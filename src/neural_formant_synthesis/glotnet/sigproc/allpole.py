import torch
import torch.nn.functional as F

def allpole(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    All-pole filter
    :param x: input signal,
        shape (B, C, T) (batch, channels, timesteps)
    :param a: filter coefficients (denominator),
        shape (C, N, T) (channels, num taps, timesteps)
    :return: filtered signal
        shape (B, C, T) (batch, channels, timesteps)
    """
    y = torch.zeros_like(x)

    a_normalized = a / a[:, 0:1, :]

    # filter order
    p = a.shape[1] - 1

    # filter coefficients
    a1 = a_normalized[:, 1:, :]

    # flip coefficients
    a1 = torch.flip(a1, [1])

    # zero pad y by filter order
    y = torch.nn.functional.pad(y, (p, 0))

    # filter
    for i in range(p, y.shape[-1]):
        y[..., i] = x[..., i - p] - \
            torch.sum(a1[..., i - p] * y[..., i - p:i], dim=-1)

    return y[..., p:]

class AllPoleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        y = allpole(x, a)
        ctx.save_for_backward(y, x, a)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, x, a = ctx.saved_tensors
        dx = da = None

        n_batch = x.size(0)
        n_channels = x.size(1)
        p = a.size(1) - 1
        T = dy.size(-1)

        # filter or
        dyda = allpole(-y, a)
        dyda = torch.nn.functional.pad(dyda, (p, 0))

        # da = torch.zeros_like(a)
        # for i in range(0, T):
        #     for j in range(0, p):
        #         da[:, p, i] = dyda[..., i:i+T] * dy
        # da = da.flip([1])


        da = F.conv1d(
            dyda.view(1, n_batch * n_channels, -1),
            dy.view(n_batch * n_channels, 1, -1),
            groups=n_batch * n_channels).view(n_batch, n_channels, -1).sum(0).flip(1)
        
        dx = allpole(dy.flip(-1), a.flip(-1)).flip(-1)

        return dx, da

class AllPole(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, a):
        return AllPoleFunction.apply(x, a)

