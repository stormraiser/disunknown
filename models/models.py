import math, random, itertools

import torch
from torch import nn
import torch.nn.functional as F
import modules

def get_paddings(size, num_levels):
	paddings = []
	for i in range(num_levels - 1):
		if size % 4 == 2:
			paddings.append(1)
			size = size // 2 + 1
		else:
			paddings.append(0)
			size //= 2
	if num_levels > 0:
		paddings.append(0)
		size //= 2
	return size, paddings

class EncoderBase(nn.Module):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers):
		super().__init__()

		num_levels = len(conv_channels)
		self.top_size, paddings = get_paddings(in_size, num_levels)
		top_channels = conv_channels[-1] if len(conv_channels) > 0 else in_channels

		conv = []
		for i in range(num_levels):
			if i == 0:
				conv.append(modules.WNConv2d(in_channels, conv_channels[i], 6, 2, 2 + paddings[0]))
				conv.append(modules.WNPReLU(conv_channels[i]))
			else:
				conv.append(modules.MiniResBlock(conv_channels[i - 1], conv_channels[i], 2, paddings[i], 1 / (i + 1)))
			for j in range(1, conv_layers[i]):
				conv.append(modules.MiniResBlock(conv_channels[i], conv_channels[i]))
		self.conv = nn.Sequential(*conv)

		self.fc = nn.Sequential(
			modules.WNLinear(top_channels * self.top_size * self.top_size, fc_features),
			modules.WNPReLU(fc_features),
			*[modules.MiniFCResBlock(fc_features) for i in range(fc_layers)]
		)

	def forward(self, input):
		return self.fc(self.conv(input).reshape(input.size(0), -1))

class Encoder(EncoderBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, code_size, zero_init = False):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers)

		self.mean_maps = nn.ModuleList()
		self.pstd_maps = nn.ModuleList()

		for m in code_size:
			self.mean_maps.append(modules.WNLinear(fc_features, m, affine = True))
			self.pstd_maps.append(modules.WNLinear(fc_features, m, affine = True))

		if zero_init:
			init_pstd = math.log(math.e - 1)
			with torch.no_grad():
				for module in self.mean_maps:
					module.scale.zero_()
					module.bias.zero_()
				for module in self.pstd_maps:
					module.scale.zero_()
					module.bias.fill_(init_pstd)

	def forward(self, input):
		base_output = super().forward(input)
		return [(mean_map(base_output), F.softplus(pstd_map(base_output))) for mean_map, pstd_map in zip(self.mean_maps, self.pstd_maps)]

class DiscriminatorBase(EncoderBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers)

	def forward(self, input, paired = None, random_offset = False, reflect = False):
		if random_offset:
			max_offset = input.size(2) // 4
			dx = [random.randrange(max_offset) + random.randrange(max_offset) for k in range(input.size(0))]
			dy = [random.randrange(max_offset) + random.randrange(max_offset) for k in range(input.size(0))]
			input_pad = F.pad(input, (max_offset, max_offset, max_offset, max_offset), mode = 'constant', value = 0)
			input = torch.stack([input_pad[k, :, dy[k] : dy[k] + input.size(2), dx[k] : dx[k] + input.size(3)] for k in range(input.size(0))], 0)
			if paired is not None:
				paired_pad = F.pad(paired, (max_offset, max_offset, max_offset, max_offset), mode = 'constant', value = 0)
				paired = torch.stack([paired_pad[k, :, dy[k] : dy[k] + paired.size(2), dx[k] : dx[k] + paired.size(3)] for k in range(paired.size(0))], 0)

		if paired is None:
			return super().forward(input)
		elif reflect:
			conv_output = self.conv(input).reshape(input.size(0), -1)
			paired_conv_output = self.conv(paired).reshape(input.size(0), -1)
			return self.fc(conv_output), self.fc(paired_conv_output), self.fc(conv_output * 2 - paired_conv_output)
		else:
			return super().forward(input), super().forward(paired), None

class Discriminator(DiscriminatorBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers)

		self.dis = modules.WNLinear(fc_features, 1, affine = True)

	def forward(self, input, paired = None, random_offset = False, reflect = False):
		if paired is None:
			base_output = super().forward(input, None, random_offset)
			return self.dis(base_output).squeeze(1)
		else:
			base_output, paired_base_output, reflect_base_output = super().forward(input, paired, random_offset, reflect)
			return self.dis(base_output).squeeze(1), self.dis(paired_base_output).squeeze(1), self.dis(reflect_base_output).squeeze(1) if reflect else None

class ClassifierBase(DiscriminatorBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, nclass):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers)

		self.cla = nn.ModuleList()
		for m in nclass:
			self.cla.append(modules.WNLinear(fc_features, m, affine = True))

	def forward(self, input, paired = None, random_offset = False, reflect = False):
		if paired is None:
			base_output = super().forward(input, None, random_offset)
			return [module(base_output) for module in self.cla]
		else:
			base_output, paired_base_output, reflect_base_output = super().forward(input, paired, random_offset, reflect)
			return [module(base_output) for module in self.cla], [module(paired_base_output) for module in self.cla], [module(reflect_base_output) for module in self.cla] if reflect else None

class Stage1ImageClassifier(ClassifierBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, nclass):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, nclass)

	def forward(self, input):
		base_output = super().forward(input)
		return [F.log_softmax(t, 1) for t in base_output]

class Stage1CodeClassifier(nn.Module):

	def __init__(self, code_size, fc_features, fc_layers, nclass):
		super().__init__()

		self.fc = nn.Sequential(
			modules.WNLinear(code_size * 2, fc_features),
			modules.WNPReLU(fc_features),
			*[modules.MiniFCResBlock(fc_features) for i in range(fc_layers)]
		)

		self.cla = nn.ModuleList()
		for m in nclass:
			self.cla.append(modules.WNLinear(fc_features, m, affine = True))

	def forward(self, input, mask, log_softmax = True):
		base_output = self.fc(torch.cat((input, 1 - mask), 1))
		if log_softmax:
			return [F.log_softmax(module(base_output), 1) for module in self.cla]
		else:
			return [module(base_output) for module in self.cla]

class Stage1CompoundClassifier(nn.Module):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, code_size, code_cla_features, code_cla_layers, nclass):
		super().__init__()

		self.img_cla = ClassifierBase(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, nclass)
		self.code_cla = Stage1CodeClassifier(code_size, code_cla_features, code_cla_layers, nclass)

	def forward(self, img, code, mask):
		img_cla_output = self.img_cla(img)
		code_cla_output = self.code_cla(code, mask, log_softmax = False)
		return [F.log_softmax(t1 + t2, 1) for t1, t2 in zip(img_cla_output, code_cla_output)]

class Stage2Classifier(ClassifierBase):

	def __init__(self, in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, nclass):
		super().__init__(in_size, in_channels, conv_channels, conv_layers, fc_features, fc_layers, [m + 1 for m in nclass])

	def forward(self, input, paired = None, random_offset = False, reflect = False, dummy_fake = False):
		if paired is None:
			base_output = super().forward(input, None, random_offset)
			return [F.log_softmax(t, 1)[:, :-1] for t in base_output] if dummy_fake else [F.log_softmax(t[:, :-1], 1) for t in base_output]
		else:
			base_output, paired_base_output, reflect_base_output = super().forward(input, paired, random_offset, reflect)
			return [F.log_softmax(t, 1)[:, :-1] for t in base_output], [F.log_softmax(t, 1)[:, :-1] for t in paired_base_output], [F.log_softmax(t, 1)[:, :-1] for t in reflect_base_output] if reflect else None

class Generator(nn.Module):

	def __init__(self, out_size, out_channels, conv_channels, conv_layers, fc_features, fc_layers, code_size):
		super().__init__()

		self.num_parts = len(code_size)
		num_levels = len(conv_channels)
		top_size, paddings = get_paddings(out_size, num_levels)
		has_conv = (num_levels > 0)
		top_channels = conv_channels[-1] if has_conv else out_channels
		self.top_shape = (-1, top_channels, top_size, top_size)

		self.fc1 = nn.Sequential(
			modules.WNLinear(sum(code_size) * 2, fc_features),
			*[modules.MiniFCResBlock(fc_features) for i in range(fc_layers)]
		)

		self.pstd_map = modules.WNLinear(fc_features, 1, affine = True)
		if has_conv:
			self.fc2 = nn.Sequential(
				modules.WNLinear(fc_features, top_channels * top_size * top_size),
				modules.WNPReLU(top_channels * top_size * top_size)
			)
		else:
			self.fc2 = nn.Sequential(
				modules.WNLinear(fc_features, top_channels * top_size * top_size, affine = True),
				nn.Tanh()
			)

		conv = []
		for i in range(num_levels - 1, -1, -1):
			for j in range(1, conv_layers[i]):
				conv.append(modules.MiniResBlock(conv_channels[i], conv_channels[i]))
			if i == 0:
				conv.append(modules.WNConvTranspose2d(conv_channels[i], out_channels, 6, 2, 2 + paddings[0], affine = True))
				conv.append(nn.Tanh())
			else:
				conv.append(modules.MiniResBlock(conv_channels[i], conv_channels[i - 1], -2, paddings[i], 1 / (num_levels - i + 1)))

		self.conv = nn.Sequential(*conv)

	def conv_params(self):
		return self.conv.parameters()

	def fc_params(self):
		return itertools.chain(self.fc1.parameters(), self.fc2.parameters(), self.pstd_map.parameters())

	def forward(self, *args):
		code = list(args[:self.num_parts])
		mask = list(args[self.num_parts:]) + [None] * (self.num_parts * 2 - len(args))
		mask = [(1 - m) if m is not None else torch.zeros_like(c) for c, m in zip(code, mask)]
		last = self.fc1(torch.cat(code + mask, 1))
		pstd = self.pstd_map(last).squeeze(1)
		return self.conv(self.fc2(last).reshape(self.top_shape)), F.softplus(pstd)

class LabelEmbedder(nn.Module):

	def __init__(self, nclass, code_size, class_freq = None, special_init = None):
		super().__init__()

		if class_freq is None:
			class_freq = torch.ones(nclass) / nclass
		self.mean = nn.Parameter(torch.zeros(nclass, code_size))
		self.lstd = nn.Parameter(torch.zeros(nclass, code_size))
		with torch.no_grad():
			if special_init == 'linear':
				self.mean[:, 0].copy_(torch.linspace(-1 + 1 / nclass, 1 - 1 / nclass, nclass))
				self.lstd[:, 0].fill_(math.log(1 / nclass))
			elif special_init == 'circle':
				angles = torch.linspace(0, math.pi * 2, nclass + 1)[:nclass]
				self.mean[:, 0].copy_(angles.cos())
				self.mean[:, 1].copy_(angles.sin())
				self.lstd[:, :2].fill_(math.log(math.sin(math.pi / nclass)))
		self.register_buffer('class_freq', class_freq)

	def stats(self):
		full_mean = self.class_freq @ self.mean
		full_var = self.class_freq @ self.mean.pow(2) - full_mean.pow(2)
		full_ev = self.class_freq @ self.lstd.exp().pow(2)
		full_std = (full_var + full_ev).clamp(min = 1e-8).sqrt()
		var_ratio = full_var / (full_var + full_ev)
		return full_mean, full_std, var_ratio

	def forward(self, input):
		full_mean, full_std, _ = self.stats()
		return (self.mean[input] - full_mean) / full_std, self.lstd[input].exp() / full_std

	def get_all(self):
		full_mean, full_std, _ = self.stats()
		return (self.mean - full_mean) / full_std, self.lstd.exp() / full_std
		
	def code_loss(self):
		full_mean, full_std, _ = self.stats()
		mean = (self.mean - full_mean) / full_std
		std = self.lstd.exp() / full_std
		return self.class_freq @ ((mean.pow(2) + std.pow(2)) * 0.5 - std.log() - 0.5)

class MultiLabelEmbedder(nn.Module):

	def __init__(self, nclass, code_size, class_freq = None, special_init = None):
		super().__init__()
		self.module_list = nn.ModuleList()
		for i in range(len(nclass)):
			self.module_list.append(LabelEmbedder(
				nclass[i], code_size[i], None if class_freq is None else class_freq[i], None if special_init is None else special_init[i]))

	def forward(self, input):
		return [module(t) for module, t in zip(self.module_list, input.t())]

	def get_all(self):
		return [module.get_all() for module in self.module_list]

	def get_var_ratio(self):
		return [module.stats()[2] for module in self.module_list]

	def code_loss(self):
		return [module.code_loss() for module in self.module_list]