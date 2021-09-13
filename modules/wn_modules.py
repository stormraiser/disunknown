import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-8

class WNConv2d(nn.Conv2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, affine = False):
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, affine)
		self.weight.data.div_(self.weight_norm())
		self.affine = affine
		if affine:
			self.scale = nn.Parameter(torch.ones(out_channels))

	def weight_norm(self):
		return self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).sum(1, keepdim = True).clamp(min = eps).sqrt()

	def forward(self, input):
		weight = self.weight / self.weight_norm()
		if self.affine:
			weight = weight * self.scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)
		return F.conv2d(input, weight, self.bias, self.stride, self.padding)

class WNConvTranspose2d(nn.ConvTranspose2d):

	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, affine = False):
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, 0, 1, affine, 1)
		self.norm_scale = (self.stride[0] * self.stride[1]) ** 0.5
		self.weight.data.div_(self.weight_norm())
		self.affine = affine
		if affine:
			self.scale = nn.Parameter(torch.ones(out_channels))

	def weight_norm(self):
		return self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).sum(0, keepdim = True).clamp(min = eps).sqrt().div(self.norm_scale)

	def forward(self, input):
		weight = self.weight / self.weight_norm()
		if self.affine:
			weight = weight * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
		return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding)

class WNLinear(nn.Linear):

	def __init__(self, in_features, out_features, affine = False):
		super().__init__(in_features, out_features, affine)
		self.weight.data.div_(self.weight_norm())
		self.affine = affine
		if affine:
			self.scale = nn.Parameter(torch.ones(out_features))

	def weight_norm(self):
		return self.weight.pow(2).sum(1, keepdim = True).clamp(min = eps).sqrt()

	def forward(self, input):
		weight = self.weight / self.weight_norm()
		if self.affine:
			weight = weight * self.scale.unsqueeze(1)
		return F.linear(input, weight, self.bias)

class WNAdd(nn.Module):

	def __init__(self, num_features, init1, init2):
		super().__init__()
		t = (init1 ** 2 + init2 ** 2) ** 0.5
		self.scale1 = nn.Parameter(torch.full((num_features,), init1 / t))
		self.scale2 = nn.Parameter(torch.full((num_features,), init2 / t))

	def weight_norm(self):
		return (self.scale1.pow(2) + self.scale2.pow(2)).clamp(min = eps).sqrt()

	def forward(self, input1, input2):
		t = self.weight_norm()
		scale1 = self.scale1.div(t).reshape(1, self.scale1.size(0), *((1,) * (input1.dim() - 2)))
		scale2 = self.scale2.div(t).reshape(1, self.scale2.size(0), *((1,) * (input2.dim() - 2)))
		return input1 * scale1 + input2 * scale2
