import torch
from torch import nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
    
    # Complex Multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        """
        # x.shape == [batch_size, in_channels, number of grid points]

        Implementation of the forward method by:
        1) Compute Fourier coefficients
        2) Multiply relevant Fourier modes
        3) Transform the data to physical space
        """
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # x.size(-1):       Returns the size of the last dimension of the tensor x, which is the number of grid points!
        # x.size(-1) // 2:  The division by half for the Fourier modes, is effectively done because only half of the 
        #                   Fourier Coefficients are stored in the output of the real-valued FFt. This is due to the 
        #                   conjugate symmetry property of the Fourier transform for real-valued inputs.
        # x.size(-1)//2 + 1: The zero frequency component is included, thus adding 1 more is necessary!
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:,:,:self.modes1] = self.compl_mul1d(x_ft[:,:,:self.modes1], self.weights1)
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x