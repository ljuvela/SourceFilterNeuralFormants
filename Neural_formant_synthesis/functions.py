import os
import numpy as np
import torch
import pyworld
import warnings

def tilt_levinson(acorr):
    """
    Calculate spectral tilt from predictor coefficient of first order allpole calculated from signal autocorrelation.
    Args:
        acorr: Autocorrelation of the signal.
    Returns:
        Spectral tilt value.
    """

    # Calculate allpole coefficients
    a,_ = levinson(acorr, 1)

    # Calculate spectral tilt
    tilt = a[:,1]

    return tilt

def spectral_centroid(x, sr):
    """
    Extract spectral centroid as weighted ratio of spectral bands.
    Args:
        x: Input frames with shape (..., n_frames, frame_length)
        sr: Sampling rate
    Returns:
        1D tensor with the values of estimated spectral centroid for each frame
    """

    mag_spec = torch.abs(torch.fft.rfft(x, dim=-1)) 
    length = x.size(-1)
    freqs = torch.abs(torch.fft.fftfreq(length, 1.0/sr)[:length//2+1]) 
    centroid = torch.sum(mag_spec*freqs, dim = -1) / torch.sum(mag_spec, dim = -1) 
    return centroid / ( sr / 2)

def frame_energy(x):
    """
    Calculate frame energy.
    Args:
        x: Input frames. size (..., n_frames, frame_length)
    Returns:
        1D tensor with energies for each frame.  Size(n_frames,)
    """

    energy = torch.sum(torch.square(torch.abs(x)), dim = -1)

    return energy

def levinson(R, M):
    """ Levinson-Durbin method for converting autocorrelation to predictor polynomial
    Args:
        R: autocorrelation tensor, shape=(..., M) 
        M: filter polynomial order     
    Returns:
        A: filter predictor polynomial tensor, shape=(..., M)
    Note:
        R can contain more lags than M, but R[..., 0:M] are required 
    """

    # Normalize autocorrelation and add white noise correction
    R = R / R[..., 0:1]
    R[..., 0:1] = R[..., 0:1] + 0.001

    E = R[..., 0:1]
    L = torch.cat([torch.ones_like(R[..., 0:1]),
                   torch.zeros_like(R[..., 0:M])], dim=-1)
    L_prev = L
    rcoeff = torch.zeros_like(L)
    for p in torch.arange(0, M):
        K = torch.sum(L_prev[..., 0:p+1] * R[..., 1:p+2], dim=-1, keepdim=True) / E
        rcoeff[...,p:p+1] = K
        if (torch.any(torch.abs(K) > 1)):
            print(torch.argmax(torch.abs(K)))
            print(R[torch.argmax(torch.abs(K)), 1:M])
            raise ValueError('Reflection coeff bigger than 1')

        pad = torch.clamp(M-p-1, min=0)
        if p == 0:
            L = torch.cat([-1.0*K,
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        else:
            L = torch.cat([-1.0*K,
                           L_prev[..., 0:p] - 1.0*K *
                           torch.flip(L_prev[..., 0:p], dims=[-1]),
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        L_prev = L
        E = E * (1.0 - K ** 2)  # % order-p mean-square error
    L = torch.flip(L, dims=[-1])  # flip zero delay to zero:th index
    return L, rcoeff

def root_to_formant(roots, sr, max_formants = 5):
    """
    Extract formant frequencies from allpole roots.
    Args:
        roots: Tensor containing roots of the polynomial.
        sr: Sampling rate.
        max_formants: Maximum number of formants to search.
    Returns:
        Tensor containing formant frequencies.
    """

    freq_tolerance = 10.0 / (sr / (2.0 * np.pi))

    phases = torch.angle(roots)
    phases = torch.where(phases < 0, phases + 2 * np.pi, phases)

    phases_sort,_ = torch.sort(phases, dim = -1, descending = False)
    
    phases_slice = phases_sort[...,1:max_formants+1]
    phases_sort = phases_sort[...,:max_formants]
    
    condition = (phases_sort[...,0:1] > freq_tolerance).repeat(1,max_formants) #Use expand instead of repeat

    phase_select = torch.where(condition, phases_sort, phases_slice)
    formants = phase_select * (sr / (2 * np.pi))

    return formants

def interpolate_unvoiced(pitch, voicing_flag = None):
    """
    Fill unvoiced regions via linear interpolation
    @Pablo: Function copied from PENN's repository to allow input of voicing flag array.
    I haven't found any robust implementations for np.interp in pytorch.
    With a differentiable version, we could add this functionality to the formant extractor.
    """
    if voicing_flag is None:
        unvoiced = pitch == 0
    else:
        unvoiced = ~voicing_flag

    # Ignore warning of log setting unvoiced regions (zeros) to nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Pitch is linear in base-2 log-space
        pitch = np.log2(pitch)

    try:

        # Interpolate
        pitch[unvoiced] = np.interp(
            np.where(unvoiced)[0],
            np.where(~unvoiced)[0],
            pitch[~unvoiced])

    except ValueError:

        # Allow all unvoiced
        pass

    return 2 ** pitch, ~unvoiced

def pitch_extraction(x, sr, window_samples, step_samples, fmin = 50, fmax = 500):
    """
    Extract pitch using pyworld.
    Params:
        x: audio signal
        sr: sample rate
        window_samples: window size in samples
        step_samples: step size in samples
        fmin: minimum pitch frequency
        fmax: maximum pitch frequency
    Returns:
        pitch: tensor with pitch values
        voicing_flag: tensor with voicing flags
    """
    # Convert to numpy
    audio = x.numpy().squeeze().astype(np.float64)

    hopsize = float(step_samples) / sr

    # Get pitch
    pitch, times  = pyworld.dio(
        audio[window_samples // 2:-window_samples // 2],
        sr,
        fmin,
        fmax,
        frame_period=1000 * hopsize)

    # Refine pitch
    pitch = pyworld.stonemask(
        audio,
        pitch,
        times,
        sr)

    # Interpolate unvoiced tokens
    pitch, voicing_flag = interpolate_unvoiced(pitch)

    # Convert to torch
    return torch.from_numpy(pitch)[None], torch.tensor(voicing_flag, dtype = torch.int)