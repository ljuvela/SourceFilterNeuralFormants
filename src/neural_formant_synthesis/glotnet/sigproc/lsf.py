import torch
import torch.nn.functional as F

def conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Convolution of two signals
    
    Args:
        x: First signal (batch, channels, time_x)
        y: Second signal (batch, channels, time_y)

    Returns:
        torch.Tensor: Convolution (batch, channels, time_x + time_y - 1) 
    
    """

    # if x.shape[0] != y.shape[0]:
    #     raise ValueError("x and y must have same batch size")
    # if x.shape[1] != y.shape[1]:
    #     raise ValueError("x and y must have same number of channels")
    
    length = x.shape[-1] + y.shape[-1] - 1

    X = torch.fft.rfft(x, n=length, dim=-1)
    Y = torch.fft.rfft(y, n=length, dim=-1)
    Z = X * Y
    z = torch.fft.irfft(Z, dim=-1, n=length)

    return z

def lsf2poly(lsf: torch.Tensor) -> torch.Tensor:
    """ Line Spectral Frequencies to Polynomial Coefficients

    Args:
        lsf (torch.Tensor): Line spectral frequencies (batch, time, order)

    Returns:
        poly (torch.Tensor): Polynomial coefficients (batch, time, order+1)


    References:
        https://github.com/cokelaer/spectrum/blob/master/src/spectrum/linear_prediction.py
    """

    if lsf.min() < 0:
        raise ValueError("lsf must be non-negative")
    if lsf.max() > torch.pi:
        raise ValueError("lsf must be less than pi")

    order = lsf.shape[-1]

    lsf, _ = torch.sort(lsf, dim=-1)

    # split to P and Q
    wP = lsf[:, :, ::2].unsqueeze(-1)
    wQ = lsf[:, :, 1::2].unsqueeze(-1)

    P_len = wP.shape[-2]
    Q_len = wQ.shape[-2]

    # compute conjugate pair polynomials
    pad = (1,1,0,0)
    Pi = F.pad(-2.0*torch.cos(wP), pad, mode='constant', value=1.0)
    Qi = F.pad(-2.0*torch.cos(wQ), pad, mode='constant', value=1.0)

    # Pi = torch.cat(1.0, -2 * torch.cos(wP), 1.0)
    # Qi = torch.cat(1.0, -2 * torch.cos(wQ), 1.0)

    # construct polynomials
    P = Pi[:,:, 0, :]
    for i in range(1, P_len):
        P = conv(P, Pi[:, :, i, :])

    Q = Qi[:, :, 0, :]
    for i in range(1, Q_len):
        Q = conv(Q, Qi[:, :, i, :])

    # add trivial zeros
    if order % 2 == 0:
        # P = conv(P, torch.tensor([1.0, -1.0]).reshape(1, 1, -1))
        # Q = conv(Q, torch.tensor([1.0, 1.0]).reshape(1, 1, -1))
        P = conv(P, torch.tensor([1.0, 1.0]).reshape(1, 1, -1))
        Q = conv(Q, torch.tensor([-1.0, 1.0]).reshape(1, 1, -1))
    else:
        # Q = conv(Q, torch.tensor([1.0, 0.0, -1.0]).reshape(1, 1, -1))
        Q = conv(Q, torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, -1))

    # compute polynomial coefficients
    A = 0.5 * (P + Q)

    return A[:, :, 1:].flip(-1)


    