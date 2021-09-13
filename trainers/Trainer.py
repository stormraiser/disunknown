import os, os.path

import torch, torchvision

from .Stage1Trainer import Stage1Trainer
from .Stage2Trainer import Stage2Trainer
from .Stage1AllUnknownTrainer import Stage1AllUnknownTrainer
from .Stage2AllUnknownTrainer import Stage2AllUnknownTrainer
from .Stage1AllLabeledTrainer import Stage1AllLabeledTrainer
from .Stage2AllLabeledTrainer import Stage2AllLabeledTrainer
from .ClassifierTrainer import ClassifierTrainer
from .LabeledEncoderTrainer import LabeledEncoderTrainer

from .utils import *

stages = ['stage1', 'classifier', 'stage2', 'lenc']

class Trainer:

	def make_stage_config(self, stage):
		stage_config = {}
		stage_config.update(self.config)
		stage_config.update(self.stage_config[stage])
		stage_config['stage'] = stage
		stage_config['start_iter'] = self.start_iter if stage == self.start_stage else 0
		return stage_config

	def __init__(self, meta_config):
		for key, value in meta_config.items():
			self.__dict__[key] = value

		for subfolder in ['checkpoints', 'log', 'samples', 'plots', 'trained_models']:
			os.makedirs(os.path.join(self.config['save_path'], subfolder), exist_ok = True)

		if self.start_stage is None:
			f = os.path.join(self.config['save_path'], 'last_checkpoint')
			if os.path.exists(f):
				with open(f) as file:
					self.start_stage, start_iter = file.read().split()
					self.start_iter = int(start_iter)
			else:
				self.start_stage = 'stage1'
				self.start_iter = 0

		if not self.has_unknown:
			self.stage1_class = Stage1AllLabeledTrainer
			self.stage2_class = Stage2AllLabeledTrainer
		elif not self.has_labeled:
			self.stage1_class = Stage1AllUnknownTrainer
			self.stage2_class = Stage2AllUnknownTrainer
		else:
			self.stage1_class = Stage1Trainer
			self.stage2_class = Stage2Trainer

	def run(self):
		start_stage_id = stages.index(self.start_stage)

		if start_stage_id <= 0:
			self.stage1_class(self.make_stage_config('stage1')).run()
		if start_stage_id <= 1 and self.has_labeled:
			ClassifierTrainer(self.make_stage_config('classifier')).run()
		if start_stage_id <= 2:
			self.stage2_class(self.make_stage_config('stage2')).run()
		if self.has_lenc_stage:
			LabeledEncoderTrainer(self.make_stage_config('lenc')).run()
