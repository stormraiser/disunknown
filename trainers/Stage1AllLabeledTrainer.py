import os.path

import torch, torchvision
from torch import optim, autograd
import torch.nn.functional as F

import models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class Stage1AllLabeledTrainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		model_config = (self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features)

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, self.labeled_size)
		self.G.to(self.device)
		self.add_model('G', self.G)

		self.B = models.MultiLabelEmbedder(self.nclass, self.labeled_size, self.class_freq, self.labeled_init)
		self.B.to(self.device)
		self.add_model('B', self.B)

		self.optimizer = optim.Adam([
			{'params': self.G.parameters()},
			{'params': self.B.parameters(), 'lr': self.emb_lr}
		], lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		if self.nested_dropout:
			self.lprob = [torch.tensor(t, device = self.device) for t in self.labeled_keep_prob]
		else:
			self.lprob = [torch.ones(t, device = self.device) for t in self.labeled_size]

		if self.start_iter == 0:
			test_set = self.dataset_class(self.data_path, 'test', self.labeled_factors, self.transform, **self.dataset_args)
			test_loader = torch.utils.data.DataLoader(test_set, batch_size = self.sample_row * self.sample_col, shuffle = True)
			self.test_image, self.test_label = next(iter(test_loader))
			torch.save(self.test_image, os.path.join(self.save_path, 'samples', 'test_image.pt'))
			torch.save(self.test_label, os.path.join(self.save_path, 'samples', 'test_label.pt'))
		else:
			self.test_image = torch.load(os.path.join(self.save_path, 'samples', 'test_image.pt'))
			self.test_label = torch.load(os.path.join(self.save_path, 'samples', 'test_label.pt'))

		self.sample_padding = max(self.image_size // 64, 1) * 2
		t = (self.test_image[:, :, 0].mean() + self.test_image[:, :, -1].mean() + self.test_image[:, :, :, 0].mean() + self.test_image[:, :, :, -1].mean()).item() / 4
		self.border_value = (t + 1) / 2

		if self.start_iter == 0:
			torchvision.utils.save_image((self.test_image + 1) / 2, os.path.join(self.save_path, 'samples', 'test_image.jpg'), nrow = self.sample_col, padding = self.sample_padding, pad_value = 255 if self.border_value < 0.5 else 0)
			self.visualize()

		self.add_periodic_action(self.visualize, self.sample_interval)

		if len(self.plot_config) > 0:
			from .plot_tools import plot_one_embedding

			for config in self.plot_config:
				if 'embedding_factor' in config:
					self.plot_one_embedding = plot_one_embedding
					self.add_periodic_action(self.plot_embedding, self.plot_embedding_interval)
					if self.start_iter == 0:
						self.plot_embedding()
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
		with torch.no_grad():
			for label in self.test_label.split(self.batch_size):
				lcode = [t[0] for t in self.B(label.to(self.device))]
				rec = self.G(*lcode)[0].cpu()
				output.append(rec)
			output = torch.cat(output, 0).add(1).div(2).clamp(min = 0, max = 1)
		filename = os.path.join(self.save_path, 'samples', 'stage1_{0}.jpg'.format(self.current_iter))
		torchvision.utils.save_image(output, filename, nrow = self.sample_col, padding = self.sample_padding, pad_value = 255 if self.border_value < 0.5 else 0)

	def iter_func(self):
		lr_factor = max(min((self.current_iter - self.emb_freeze) / self.lr_ramp, 1), 0)
		self.optimizer.param_groups[1]['lr'] = self.emb_lr * lr_factor
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor

		image, label = self.next_batch()

		self.optimizer.zero_grad()

		lparam = self.B(label)
		ret = [drop_and_sample(*t, p) for t, p in zip(lparam, self.lprob)]
		lmask = [t[0] for t in ret]
		lcode = [t[1] for t in ret]
		ret = self.B.code_loss()
		lcode_loss = sum([t.mul(p).sum() for t, p in zip(ret, self.lprob)])
		lcode_loss_log = sum([t.sum().item() for t in ret])

		rec, rec_std = self.G(*lcode, *lmask)
		rec_loss = rec_loss_func(rec, image, rec_std)
		(rec_loss * self.rec_weight + lcode_loss * self.lcode_weight).backward()

		if self.test_rec:
			with torch.no_grad():
				rec_test, rec_test_std = self.G(*[t[0] for t in lparam])
				rec_test_loss = rec_loss_func(rec_test, image, rec_test_std)

		self.optimizer.step()

		self.log(rec = rec_loss.item(), lcode = lcode_loss_log)
		if self.test_rec:
			self.log(rtest = rec_test_loss.item())

	def report_loss(self):
		self.print('s1 {iter}: rec={rec:.3f}' + ('/{rtest:.3f}' if self.test_rec else '') + ' lcode={lcode:.2f}')

	def finalize(self):
		torch.save(self.G.state_dict(), os.path.join(self.save_path, 'trained_models', 'G1.pt'))
		torch.save(self.B.state_dict(), os.path.join(self.save_path, 'trained_models', 'B1.pt'))

	def plot_embedding(self):
		lparam = self.B.get_all()
		var_ratio = self.B.get_var_ratio()

		for k, config in enumerate(self.plot_config):
			if 'embedding_factor' in config:
				code_factor_id = self.labeled_factors.index(config['embedding_factor'])
				filename = os.path.join(self.save_path, 'plots', '{0}_{1}_s1_{2}.png'.format(k, config['embedding_factor'], self.current_iter))
				self.plot_one_embedding(*lparam[code_factor_id], var_ratio[code_factor_id], config.get('dims'), config.get('colormap'), filename)
