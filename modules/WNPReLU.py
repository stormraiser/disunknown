import math
import torch
from torch import nn
import torch.nn.functional as F

def phi(x):
	return 1 / ((2 * math.pi) ** 0.5) * (-0.5 * x.pow(2)).exp()

def Phi(x):
	return 0.5 * (1 + (x / (2 ** 0.5)).erf())

class WNPReLU(nn.Module):

	def __init__(self, num_features = 1, init = 0.25):
		super().__init__()
		self.num_features = num_features
		self.pre_bias = nn.Parameter(torch.zeros(num_features))
		self.weight = nn.Parameter(torch.full((num_features,), init, dtype = torch.float32))

	def forward(self, input):
		phi_t = phi(-self.pre_bias)
		Phi_t = Phi(-self.pre_bias)
		mean = self.pre_bias + (self.weight - 1) * (Phi_t * self.pre_bias - phi_t)
		ex2 = 1 + self.pre_bias.pow(2) + (self.weight.pow(2) - 1) * (Phi_t * (1 + self.pre_bias.pow(2)) - phi_t * self.pre_bias)
		std = (ex2 - mean.pow(2)).sqrt()
		pre_bias = self.pre_bias.reshape(1, self.pre_bias.size(0), *((1,) * (input.dim() - 2)))
		mean = mean.reshape(1, mean.size(0), *((1,) * (input.dim() - 2)))
		std = std.reshape(1, std.size(0), *((1,) * (input.dim() - 2)))
		return (F.prelu(input + pre_bias, self.weight) - mean) / std
