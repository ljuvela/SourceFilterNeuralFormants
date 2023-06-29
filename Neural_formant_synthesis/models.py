import torch

class LAR_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(k_hat, k_ref):
        """
        Calculate loss function as the error in Log Area Ratio domain from reflection coefficients as input.
        Params:
            k_hat: Estimated reflection coefficients with shape (batch_size, num_frames, num_coeffs)
            k_ref: Reference reflection coefficients with shape (batch_size, num_frames, num_coeffs)
        Return:
            Loss value
        """
        batch_size = k_hat.size(0)

        lar_hat = torch.log(torch.divide(1 - k_hat, 1 + k_hat))
        lar_ref = torch.log(torch.divide(1 - k_ref, 1 + k_ref))

        abs_err = torch.sum(torch.square(lar_ref - lar_hat))

        loss = torch.divide(abs_err, batch_size)

        return loss