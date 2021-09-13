import os.path

import torch
from torch import optim
import torch.nn.functional as F

import models
from .StageTrainer import StageTrainer

class ClassifierTrainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		self.R = models.Stage2Classifier(self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features, self.dis_cla_fc_layers, self.nclass)
		self.R.to(self.device)
		self.add_model('R', self.R)

		self.optimizer = optim.Adam(self.R.parameters(), lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

	def next_batch(self):
		try:
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			batch = next(self.data_iter)

		image, label = batch

		return image.to(self.device), label.to(self.device)

	def iter_func(self):
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor

		image, label = self.next_batch()

		self.optimizer.zero_grad()

		cla2_output = self.R(image, random_offset = self.random_offset)
		cla2_loss = sum([F.nll_loss(c, l) for c, l in zip(cla2_output, label.t())])
		cla2_loss.backward()

		self.optimizer.step()

		self.log(cla2 = cla2_loss.item())

	def report_loss(self):
		self.print('cla2 {iter}: cla2={cla2:.2f}')

	def finalize(self):
		torch.save(self.R.state_dict(), os.path.join(self.save_path, 'trained_models', 'R1.pt'))
