import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse

import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict, build_env
from neural_formant_synthesis.third_party.hifi_gan.meldataset import mel_spectrogram
from neural_formant_synthesis.third_party.hifi_gan.models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_adversarial_loss,\
    discriminator_loss
#from neural_formant_synthesis.third_party.hifi_gan.models import discriminator_metrics
from neural_formant_synthesis.third_party.hifi_gan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint


from neural_formant_synthesis.glotnet.sigproc.lpc import LinearPredictor
from neural_formant_synthesis.glotnet.sigproc.emphasis import Emphasis

from neural_formant_synthesis.dataset import FeatureDataset_List
from neural_formant_synthesis.models import FM_Hifi_Generator, fm_config_obj, Envelope_wavenet, Envelope_conformer
from neural_formant_synthesis.models import NeuralFormantSynthesisGenerator


from neural_formant_synthesis.glotnet.sigproc.levinson import forward_levinson

import torchaudio as ta


torch.backends.cudnn.benchmark = True


def train(rank, a, h, fm_h):

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')

    # HiFi generator included in feature mapping class.
    pretrained_fm = getattr(fm_h, 'model_path', None)
    generator = NeuralFormantSynthesisGenerator(fm_config = fm_h, g_config = h,
                                  pretrained_fm = pretrained_fm,
                                  freeze_fm = pretrained_fm is not None, 
                                  device = device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Num parameters in HiFi Generator: {count_parameters(generator.hifi_generator)}")
    print(f"Num parameters in Feature Mapping model: {count_parameters(generator.feature_mapping)}")

    generator = generator.to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    pretrain_steps = 0 # always reset, never load
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        generator.load_generator_e2e_checkpoint(cp_g)
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_path = os.path.join(a.input_wavs_dir, 'train')
    trainset = FeatureDataset_List(training_path, h, sampling_rate = h.sampling_rate, 
                                   frame_size = h.win_size, hop_size = h.hop_size, shuffle = True, audio_ext = '.flac', 
                                   segment_length = 32)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        
        valid_path = os.path.join(a.input_wavs_dir, 'val')
        validset = FeatureDataset_List(valid_path, h, sampling_rate = h.sampling_rate, 
                                       frame_size = h.win_size, hop_size = h.hop_size, audio_ext = '.flac')
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))


    generator.train()
    mpd.train()
    msd.train()

    # generator = torch.compile(generator)
    # mpd = torch.compile(mpd)
    # msd = torch.compile(msd)
    for epoch in range(max(0, last_epoch), a.training_epochs):

        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()

            #size --> (Batch, features, sequence)
            x, _, y, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            x_feat = x[:,0:9,:]

            # feature_map_only = pretrain_steps < fm_h.get('pre_train_steps', 0)
            feature_map_only = steps < fm_h.get('pre_train_steps', 0)
            detach = fm_h.get('detach_feature_map_from_gan', False)
            if feature_map_only:
                mel_cond = generator.forward(x_feat, feature_map_only, detach)
            else:
                y_g_hat, mel_cond = generator.forward(x_feat, feature_map_only, detach)



            if not feature_map_only:
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft,
                                            h.num_mels, h.sampling_rate,
                                            h.hop_size, h.win_size,
                                            h.fmin, h.fmax_for_loss)

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())

                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())

                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_gen_all = 0.0

            if not feature_map_only:
                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                # Adversarial losses
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_adversarial_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_adversarial_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_all + loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_mel_fm = F.l1_loss(y_mel, mel_cond)

            loss_gen_all = loss_gen_all + loss_mel_fm

            loss_gen_all.backward()
            optim_g.step()

            if not torch.isfinite(loss_gen_all):
                raise ValueError(f"Loss value is not finite, was {loss_gen_all}")

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:

                    if feature_map_only:
                        print('Steps : {:d}, Gen Loss Total : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, time.time() - start_b))
                    else:
                        with torch.no_grad():
                            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                        print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)

                    sw.add_scalar("training/feature_mapping_mel_loss", loss_mel_fm, steps)

                    if not feature_map_only:
                        sw.add_scalar("training/mel_spec_error", mel_error, steps)

                        # Framed Discriminator losses
                        sw.add_scalar("training_gan/disc_f_r", sum(losses_disc_f_r), steps)
                        sw.add_scalar("training_gan/disc_f_g", sum(losses_disc_f_g), steps)
                        # Multiscale Discriminator losses
                        sw.add_scalar("training_gan/disc_s_r", sum(losses_disc_s_r), steps)
                        sw.add_scalar("training_gan/disc_s_g", sum(losses_disc_s_g), steps)
                        # Framed Generator losses
                        sw.add_scalar("training_gan/gen_f", sum(losses_gen_f), steps)
                        # Multiscale Generator losses
                        sw.add_scalar("training_gan/gen_s", sum(losses_gen_s), steps)
                        # Feature Matching losses
                        sw.add_scalar("training_gan/loss_fm_f", loss_fm_f, steps)
                        sw.add_scalar("training_gan/loss_fm_s", loss_fm_s, steps)


                # Validation
                if steps % a.validation_interval == 0: #and steps != 0:

                    print(f"Validation at step {steps}")
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    max_valid_batches = 100
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):

                            if j > max_valid_batches:
                                break

                            x, _, y, y_mel = batch

                            x = x.to(device)

                            x_feat = x[:,0:9,:]

                            y_g_hat, mel_cond = generator(x_feat)

                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)

                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            # TODO: calculate discriminator EER

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1
            pretrain_steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--fm_config', default='')
    parser.add_argument('--env_config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--envelope_model_pretrained', default=False, type=bool)
    parser.add_argument('--envelope_model_freeze', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    with open(a.fm_config) as f:
        data = f.read()
    json_fm_config = json.loads(data)
    fm_h = AttrDict(json_fm_config)


    build_env(a.config, 'config.json', a.checkpoint_path)
    # TODO: copy configs for feature mapping and envelope models!

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        # Skip multi-gpu implementation until supported by all the models.
        h.num_gpus = 1 #torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h, fm_h))
    else:
        train(0, a, h, fm_h)


if __name__ == '__main__':
    main()
