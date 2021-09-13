import os.path, pickle

import torch, torchvision
import data
from .utils import *

loss_smooth_alpha = 0.99

class StageTrainer:

	def __init__(self, config):
		for key, value in config.items():
			self.__dict__[key] = value
		self.current_iter = self.start_iter
		self.dataset_class = data.datasets[self.dataset_name]
		self.record_progress()

		self.transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize(self.image_size),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize([0.5] * self.image_channels, [0.5] * self.image_channels)
		])
		self.dataset = self.dataset_class(self.data_path, 'train', self.labeled_factors, self.transform, **self.dataset_args)
		self.nclass = self.dataset.nclass
		self.class_freq = [t.to(self.device) for t in self.dataset.class_freq]

		self.models = {}
		self.log_dict = {}
		self.loss_report = {}

		self.actions = []

		self.default_actions = [
			(self.report_loss, self.report_interval),
			(self.save_log, self.log_interval),
			(self.save_checkpoint, self.checkpoint_interval)
		]

	def record_progress(self):
		with open(os.path.join(self.save_path, 'last_checkpoint'), 'w') as file:
			print(self.stage, self.current_iter, file = file)

	def add_model(self, name, model):
		self.models[name] = model

	def save_checkpoint(self):
		save_dict = {}
		for name, model in self.models.items():
			save_dict[name] = model.state_dict()
		torch.save(save_dict, os.path.join(self.save_path, 'checkpoints', '{0}_{1}.pt'.format(self.stage, self.current_iter)))
		self.save_log()
		self.record_progress()

	def load_checkpoint(self):
		load_dict = torch.load(os.path.join(self.save_path, 'checkpoints', '{0}_{1}.pt'.format(self.stage, self.start_iter)), map_location = self.device)
		for name, model in self.models.items():
			model.load_state_dict(load_dict[name])

	def add_periodic_action(self, func, interval):
		self.actions.append((func, interval))

	def log(self, **kwargs):
		for name, value in kwargs.items():
			if name in self.loss_report:
				self.loss_report[name] = self.loss_report[name] * loss_smooth_alpha + value * (1 - loss_smooth_alpha)
			else:
				self.loss_report[name] = value
			if name in self.log_dict:
				self.log_dict[name].append((self.current_iter, value))
			else:
				self.log_dict[name] = [(self.current_iter, value)]

	def save_log(self):
		if len(self.log_dict) > 0:
			with open(os.path.join(self.save_path, 'log', '{0}_{1}'.format(self.stage, self.current_iter)), 'wb') as log_file:
				pickle.dump(self.log_dict, log_file)
			self.log_dict.clear()

	def print(self, format_str):
		print(format_str.format(iter = self.current_iter, **self.loss_report))

	def report_loss(self):
		pass

	def iter_func(self):
		pass

	def finalize(self):
		pass

	def run(self):
		if self.start_iter > 0:
			self.load_checkpoint()

		while self.current_iter < self.niter:
			self.current_iter += 1
			self.iter_func()

			for func, interval in self.actions + self.default_actions:
				if self.current_iter % interval == 0 or self.current_iter == self.niter:
					func()

		self.finalize()

	def save_image_pair(self, image1, image2, filename, padding, border):
		image = torch.cat((image1, torch.full((image1.size(0), image1.size(1), self.image_size, padding), border), image2), 3)
		torchvision.utils.save_image(image, filename, nrow = self.sample_col, padding = padding, pad_value = 255 if border < 0.5 else 0)
