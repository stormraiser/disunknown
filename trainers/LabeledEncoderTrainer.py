import os.path

import torch
from torch import optim
import torch.nn.functional as F

import models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class LabeledEncoderTrainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		model_config = (self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features)

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size] + self.labeled_size)
		self.G.to(self.device)
		self.G.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'G2.pt'), map_location = self.device))

		self.B = models.MultiLabelEmbedder(self.nclass, self.labeled_size, self.class_freq)
		self.B.to(self.device)
		self.B.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'B2.pt'), map_location = self.device))

		self.S = models.Encoder(*model_config, self.enc_gen_fc_layers, self.labeled_size, zero_init = True)
		self.S.to(self.device)
		self.add_model('S', self.S)

		self.optimizer = optim.Adam(self.S.parameters(), lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		self.test_image = torch.load(os.path.join(self.save_path, 'samples', 'test_image.pt'))
		if self.unknown_size > 0:
			self.test_ucode = torch.load(os.path.join(self.save_path, 'samples', 'test_ucode.pt'))
		else:
			self.test_ucode = torch.zeros(self.test_image.size(0), 2, 0)

		self.sample_padding = max(self.image_size // 64, 1) * 2
		t = (self.test_image[:, :, 0].mean() + self.test_image[:, :, -1].mean() + self.test_image[:, :, :, 0].mean() + self.test_image[:, :, :, -1].mean()).item() / 4
		self.border_value = (t + 1) / 2

		if self.start_iter == 0:
			self.visualize()
		self.add_periodic_action(self.visualize, self.sample_interval)

		if len(self.plot_config) > 0:
			from .plot_tools import plot_one_code

			for config in self.plot_config:
				if 'code_factor' in config and config['code_factor'] != 'unknown':
					self.plot_dataset = self.dataset_class(self.data_path, 'plot', self.all_factors, self.transform, **self.dataset_args)
					self.plot_one_code = plot_one_code
					self.add_periodic_action(self.plot_code, self.plot_code_interval)
					if self.start_iter == 0:
						self.plot_code()
					break

	def next_batch(self):
		try:
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			batch = next(self.data_iter)

		image, label = batch

		return image.to(self.device), label.to(self.device)

	def visualize(self):
		output = []
		output_alt = []
		test_lcode = [[] for i in range(len(self.labeled_factors))]
		with torch.no_grad():
			for image in self.test_image.split(self.batch_size):
				ret = self.S(image.to(self.device))
				for i in range(len(self.labeled_factors)):
					test_lcode[i].append(ret[i][0])
			test_lcode = [torch.cat(t, 0) for t in test_lcode]
			test_lcode_alt = [torch.cat((t[1:], t[:1]), 0) for t in test_lcode]

			for i in range((self.test_image.size(0) - 1) // self.batch_size + 1):
				ucode = self.test_ucode[i * self.batch_size : (i + 1) * self.batch_size, 0].to(self.device)
				lcode = [t[i * self.batch_size : (i + 1) * self.batch_size] for t in test_lcode]
				lcode_alt = [t[i * self.batch_size : (i + 1) * self.batch_size] for t in test_lcode_alt]
				rec = self.G(ucode, *lcode)[0].cpu()
				cross = self.G(ucode, *lcode_alt)[0].cpu()
				output.append(rec)
				output_alt.append(cross)
			output = torch.cat(output, 0).add(1).div(2).clamp(min = 0, max = 1)
			output_alt = torch.cat(output_alt, 0).add(1).div(2).clamp(min = 0, max = 1)
		filename = os.path.join(self.save_path, 'samples', 'lenc_{0}.jpg'.format(self.current_iter))
		self.save_image_pair(output, output_alt, filename, self.sample_padding, self.border_value)

	def iter_func(self):
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor

		image, label = self.next_batch()

		self.optimizer.zero_grad()

		with torch.no_grad():
			target_lparams = self.B(label)

		lparams = self.S(image)
		lenc_loss = sum([normal_kld(*t1, t2[0].detach(), t2[1].detach()) for t1, t2 in zip(lparams, target_lparams)])
		lenc_loss.backward()

		self.optimizer.step()

		self.log(lenc = lenc_loss.item())

	def report_loss(self):
		self.print('lenc {iter}: lenc={lenc:.2f}')

	def finalize(self):
		torch.save(self.S.state_dict(), os.path.join(self.save_path, 'trained_models', 'S.pt'))

		encode_dataset = self.dataset_class(self.data_path, 'train', [], self.transform, **self.dataset_args)
		encode_loader = torch.utils.data.DataLoader(encode_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		lcode_train = run_encoder(encode_loader, self.S, self.device)
		lcode_train_stats = [get_code_stats(t, self.device) for t in lcode_train]

		encode_dataset = self.dataset_class(self.data_path, 'test', [], self.transform, **self.dataset_args)
		encode_loader = torch.utils.data.DataLoader(encode_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		lcode_test = run_encoder(encode_loader, self.S, self.device)
		lcode_test_stats = [get_code_stats(t, self.device) for t in lcode_test]

		lcode_list = []
		for k in range(len(self.labeled_factors)):
			lcode_list.append({
				'train': lcode_train[k],
				'train_stats': lcode_train_stats[k],
				'test': lcode_test[k],
				'test_stats': lcode_test_stats[k]
			})
		torch.save(lcode_list, os.path.join(self.save_path, 'trained_models', 'labeled_code.pt'))

	def plot_code(self):
		plot_nclass = self.plot_dataset.nclass
		plot_loader = torch.utils.data.DataLoader(self.plot_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		lcode_plot, plot_label = run_encoder(plot_loader, self.S, self.device)
		lcode_plot_stats = [get_code_stats(t, self.device) for t in lcode_plot]

		for k, config in enumerate(self.plot_config):
			if 'code_factor' in config and config['code_factor'] != 'unknown':
				code_factor_id = self.labeled_factors.index(config['code_factor'])
				color_factor_id = self.all_factors.index(config['color_factor'])
				color = plot_label[:, color_factor_id].float() / plot_nclass[color_factor_id]
				filename = os.path.join(self.save_path, 'plots', '{0}_{1}_{2}_lenc_{3}.png'.format(k, config['code_factor'], config['color_factor'], self.current_iter))
				self.plot_one_code(lcode_plot[code_factor_id], lcode_plot_stats[code_factor_id], config.get('dims'), color, config.get('colormap'), filename)
