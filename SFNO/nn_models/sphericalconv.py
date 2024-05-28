import torch
from torch import nn
import torch_harmonics as th

class SphericalConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SphericalConv, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # transform data on an equiangular grid
        nlat = x.shape[2] # latitude
        nlon = x.shape[3] # longitude
        sht = th.RealSHT(nlat, nlon, grid="equiangular")
        isht = th.InverseRealSHT(nlat, nlon, grid="equiangular")

        # BASIS TRANSFORMATION
        # Computation of the spherical harmonic transform, taking into account the geometry of the sphere
        x_ht = sht(x)

        # Multiply relevant Fourier modes
        out_ht = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ht[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ht[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ht[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ht[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = isht(out_ht)
        return x
