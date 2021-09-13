import torch
from torch import nn
import torch.nn.functional as F

from .wn_modules import WNConv2d, WNConvTranspose2d, WNAdd, WNLinear
from .WNPReLU import WNPReLU

class UpSample2x(nn.Module):

	def __init__(self, padding):
		super().__init__()
		self.padding = padding

	def forward(self, input):
		ret = F.interpolate(input, scale_factor = 2, mode = 'nearest')
		if self.padding > 0:
			ret = ret[:, :, self.padding : -self.padding, self.padding : -self.padding]
		return ret

class MiniResBlock(nn.Module):

	def __init__(self, in_channels, out_channels, stride = 1, extra_padding = 0, residue_ratio = 0):
		super().__init__()

		if stride == 2:
			conv1 = WNConv2d(in_channels, out_channels, 4, 2, 1 + extra_padding)
		elif stride == -2:
			conv1 = WNConvTranspose2d(in_channels, out_channels, 4, 2, 1 + extra_padding)
		else:
			conv1 = WNConv2d(in_channels, out_channels, 3, 1, 1 + extra_padding)

		self.residue = nn.Sequential(
			conv1,
			WNPReLU(out_channels)
		)

		self.shortcut = nn.Sequential(
			nn.AvgPool2d(2, padding = extra_padding) if stride == 2 else nn.Sequential(),
			WNConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Sequential(),
			UpSample2x(extra_padding) if stride == -2 else nn.Sequential()
		)

		self.add = WNAdd(out_channels, (1 - residue_ratio) ** 0.5, residue_ratio ** 0.5)

	def forward(self, input):
		return self.add(self.shortcut(input), self.residue(input))

class MiniFCResBlock(nn.Module):

	def __init__(self, num_features):
		super(MiniFCResBlock, self).__init__()

		self.residue = nn.Sequential(
			WNLinear(num_features, num_features),
			WNPReLU(num_features)
		)

		self.add = WNAdd(num_features, 1, 0)

	def forward(self, input):
		return self.add(input, self.residue(input))
