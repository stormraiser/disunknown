import os.path

import torch, torchvision
from torch import optim, autograd
import torch.nn.functional as F

import models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class Stage1AllUnknownTrainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		model_config = (self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features)

		self.E = models.Encoder(*model_config, self.enc_gen_fc_layers, [self.unknown_size], zero_init = self.zero_init)
		self.E.to(self.device)
		self.add_model('E', self.E)

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size])
		self.G.to(self.device)
		self.add_model('G', self.G)

		self.optimizer = optim.Adam([
			{'params': self.E.parameters()},
			{'params': self.G.parameters()}
		], lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		if self.nested_dropout:
			self.uprob = torch.tensor(self.unknown_keep_prob, device = self.device)
		else:
			self.uprob = torch.ones(self.unknown_size, device = self.device)

		if self.start_iter == 0:
			test_set = self.dataset_class(self.data_path, 'test', [], self.transform, **self.dataset_args)
			test_loader = torch.utils.data.DataLoader(test_set, batch_size = self.sample_row * self.sample_col, shuffle = True)
			self.test_image = next(iter(test_loader))
			torch.save(self.test_image, os.path.join(self.save_path, 'samples', 'test_image.pt'))
		else:
			self.test_image = torch.load(os.path.join(self.save_path, 'samples', 'test_image.pt'))

		self.sample_padding = max(self.image_size // 64, 1) * 2
		t = (self.test_image[:, :, 0].mean() + self.test_image[:, :, -1].mean() + self.test_image[:, :, :, 0].mean() + self.test_image[:, :, :, -1].mean()).item() / 4
		self.border_value = (t + 1) / 2

		if self.start_iter == 0:
			torchvision.utils.save_image((self.test_image + 1) / 2, os.path.join(self.save_path, 'samples', 'test_image.jpg'), nrow = self.sample_col, padding = self.sample_padding, pad_value = 255 if self.border_value < 0.5 else 0)
			self.visualize()

		self.add_periodic_action(self.visualize, self.sample_interval)

		if len(self.plot_config) > 0:
			from .plot_tools import plot_one_code

			for config in self.plot_config:
				if config.get('code_factor') == 'unknown':
					self.plot_dataset = self.dataset_class(self.data_path, 'plot', self.all_factors, self.transform, **self.dataset_args)
					self.plot_one_code = plot_one_code
					self.add_periodic_action(self.plot_code, self.plot_code_interval)
					if self.start_iter == 0:
						self.plot_code()
					break

		if self.growing_dataset:
			self.add_periodic_action(self.update_dataset, self.dataset_update_interval)
			self.update_dataset()

	def update_dataset(self):
		self.dataset.update(self.current_iter)
		self.B.update_class_freq(self.dataset.class_freq)
		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)
		if hasattr(self, 'plot_dataset'):
			self.plot_dataset.update(self.current_iter)

	def next_batch(self):
		try:
			image = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			image = next(self.data_iter)

		return image.to(self.device)

	def visualize(self):
		output = []
		with torch.no_grad():
			for image in self.test_image.split(self.batch_size):
				ucode = self.E(image.to(self.device))[0][0]
				rec = self.G(ucode)[0].cpu()
				output.append(rec)
			output = torch.cat(output, 0).add(1).div(2).clamp(min = 0, max = 1)
		filename = os.path.join(self.save_path, 'samples', 'stage1_{0}.jpg'.format(self.current_iter))
		torchvision.utils.save_image(output, filename, nrow = self.sample_col, padding = self.sample_padding, pad_value = 255 if self.border_value < 0.5 else 0)

	def iter_func(self):
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor
		self.optimizer.param_groups[1]['lr'] = self.lr * lr_factor

		image = self.next_batch()

		self.optimizer.zero_grad()

		umean, ustd = self.E(image)[0]
		umask, ucode = drop_and_sample(umean, ustd, self.uprob)
		ucode_loss, ucode_loss_log, ubatch_loss, ubatch_loss_log = code_loss_func(umean, ustd, self.uprob)

		rec, rec_std = self.G(ucode, umask)
		rec_loss = rec_loss_func(rec, image, rec_std)
		(rec_loss * self.rec_weight + ucode_loss * self.ucode_weight + ubatch_loss * self.ubatch_weight).backward()

		if self.test_rec:
			with torch.no_grad():
				rec_test, rec_test_std = self.G(umean)
				rec_test_loss = rec_loss_func(rec_test, image, rec_test_std)

		self.optimizer.step()

		self.log(rec = rec_loss.item(), ucode = ucode_loss_log, ubatch = ubatch_loss_log)
		if self.test_rec:
			self.log(rtest = rec_test_loss.item())

	def report_loss(self):
		self.print('s1 {iter}: rec={rec:.3f}' + ('/{rtest:.3f}' if self.test_rec else '') + ' ucode={ucode:.2f}/{ubatch:.2f}')

	def finalize(self):
		torch.save(self.E.state_dict(), os.path.join(self.save_path, 'trained_models', 'E.pt'))
		torch.save(self.G.state_dict(), os.path.join(self.save_path, 'trained_models', 'G1.pt'))

		encode_dataset = self.dataset_class(self.data_path, 'train', [], self.transform, **self.dataset_args)
		encode_loader = torch.utils.data.DataLoader(encode_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		ucode_train = run_encoder(encode_loader, self.E, self.device)[0]
		ucode_train_stats = get_code_stats(ucode_train, self.device)

		encode_dataset = self.dataset_class(self.data_path, 'test', [], self.transform, **self.dataset_args)
		encode_loader = torch.utils.data.DataLoader(encode_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		ucode_test = run_encoder(encode_loader, self.E, self.device)[0]
		ucode_test_stats = get_code_stats(ucode_test, self.device)

		ucode_dict = {
			'train': ucode_train,
			'train_stats': ucode_train_stats,
			'test': ucode_test,
			'test_stats': ucode_test_stats
		}
		torch.save(ucode_dict, os.path.join(self.save_path, 'trained_models', 'unknown_code.pt'))

		ucode_vis = run_encoder(self.test_image.split(self.batch_size), self.E, self.device)[0]
		torch.save(ucode_vis, os.path.join(self.save_path, 'samples', 'test_ucode.pt'))

	def plot_code(self):
		plot_nclass = self.plot_dataset.nclass
		plot_loader = torch.utils.data.DataLoader(self.plot_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
		ucode_plot, plot_label = run_encoder(plot_loader, self.E, self.device)
		ucode_plot = ucode_plot[0]
		ucode_plot_stats = get_code_stats(ucode_plot, self.device)

		for k, config in enumerate(self.plot_config):
			if config.get('code_factor') == 'unknown':
				color_factor_id = self.all_factors.index(config['color_factor'])
				color = plot_label[:, color_factor_id].float() / plot_nclass[color_factor_id]
				filename = os.path.join(self.save_path, 'plots', '{0}_unknown_{1}_s1_{2}.png'.format(k, config['color_factor'], self.current_iter))
				self.plot_one_code(ucode_plot, ucode_plot_stats, config.get('dims'), color, config.get('colormap'), filename)
