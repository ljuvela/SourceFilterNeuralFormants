import torch
import torch.nn.functional as F

def toeplitz(r: torch.Tensor):
    """" Construct Toeplitz matrix """
    p = r.size(-1)   
    rr = torch.cat([r, r[..., 1:].flip(dims=(-1,))], dim=-1)
    T = [torch.roll(rr, i, dims=(-1,))[...,:p] for i in range(p)]
    return torch.stack(T, dim=-1)


def levinson(R:torch.Tensor, M:int, eps:float=1e-3) -> torch.Tensor:
    """ Levinson-Durbin method for converting autocorrelation to all-pole polynomial
    Args:
        R: autocorrelation, shape=(..., M) 
        M: filter polynomial order     
    Returns:
        A: all-pole polynomial, shape=(..., M)
    """
    # normalize R
    R = R / R[..., 0:1]
    # white noise correction
    R[..., 0] = R[..., 0] + eps

    # zero lag
    K = torch.sum(R[..., 1:2], dim=-1, keepdim=True)
    A = torch.cat([-1.0*K, torch.ones_like(R[..., 0:1])], dim=-1)
    E = 1.0 - K ** 2
    # higher lags
    for p in torch.arange(1, M):
        K = torch.sum(A[..., 0:p+1] * R[..., 1:p+2], dim=-1, keepdim=True) / E
        if K.abs().max() > 1.0:
            raise ValueError(f"Unstable filter, |K| was {K.abs().max()}")
        A = torch.cat([-1.0*K,
                        A[..., 0:p] - 1.0*K *
                        torch.flip(A[..., 0:p], dims=[-1]),
                        torch.ones_like(R[..., 0:1])], dim=-1)
        E = E * (1.0 - K ** 2)
    A = torch.flip(A, dims=[-1])
    return A


def forward_levinson(K: torch.Tensor, M: int = None) -> torch.Tensor:
    """ Forward Levinson converts reflection coefficients to all-pole polynomial

        Args:
            K: reflection coefficients, shape=(..., M) 
            M: filter polynomial order (optional, defaults to K.size(-1))
        Returns:
            A: all-pole polynomial, shape=(..., M+1)

    """
    if M is None:
        M = K.size(-1)

    A = -1.0*K[..., 0:1]
    for p in torch.arange(1, M):
        A = torch.cat([-1.0*K[..., p:p+1],
                        A[..., 0:p] - 1.0*K[..., p:p+1] * torch.flip(A[..., 0:p], dims=[-1])], dim=-1)

    A = torch.cat([A, torch.ones_like(A[..., 0:1])], dim=-1)
    A = torch.flip(A, dims=[-1]) # flip zero delay to zero:th index
    return A


def spectrum_to_allpole(spectrum:torch.Tensor, order:int, root_scale:float=1.0):
    """ Convert spectrum to all-pole filter coefficients
    
    Args:
        spectrum: power spectrum (squared magnitude), shape=(..., K)
        order: filter polynomial order

    Returns:
        a: filter predictor polynomial tensor, shape=(..., order+1)
        g: filter gain
    """
    r = torch.fft.irfft(spectrum, dim=-1)
    # add small value to diagonal to avoid singular matrix
    r[..., 0] = r[..., 0] + 1e-6 
    # all pole from autocorr
    a = levinson(r, order)

    # filter gain
    # g = torch.sqrt(torch.dot(r[:(order+1)], a))
    g = torch.sqrt(torch.sum(r[..., :(order+1)] * a, dim=-1, keepdim=True))

    # scale filter roots
    if root_scale < 1.0:
        a = a * root_scale ** torch.arange(order+1, dtype=torch.float32, device=a.device)

    return a, g