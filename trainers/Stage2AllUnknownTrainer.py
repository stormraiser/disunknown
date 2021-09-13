import os.path, itertools

import torch, torchvision
from torch import optim, autograd
import torch.nn.functional as F

import data, models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class Stage2AllUnknownTrainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		ucode_dict = torch.load(os.path.join(self.save_path, 'trained_models', 'unknown_code.pt'))
		self.ucode_dataset = torch.utils.data.TensorDataset(ucode_dict['train'])
		self.ucode_stats = ucode_dict['train_stats']
		for key in self.ucode_stats:
			self.ucode_stats[key] = self.ucode_stats[key].to(self.device)
		self.paired_dataset = data.PairedDataset(self.dataset, self.ucode_dataset)
		
		self.branch_factor = 0.5 if (self.has_rec_branch and self.random_ucode) else 1

		model_config = (self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features)

		if self.unknown_mode == 'mse':
			self.E = models.Encoder(*model_config, self.enc_gen_fc_layers, [self.unknown_size], zero_init = True)
			self.E.to(self.device)
			self.E.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'E.pt'), map_location = self.device))
			self.E.requires_grad_(False)
		else:
			self.G_fixed = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size] + self.labeled_size)
			self.G_fixed.to(self.device)
			self.G_fixed.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'G1.pt'), map_location = self.device))

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size] + self.labeled_size)
		self.G.to(self.device)
		self.add_model('G', self.G)
		if self.start_iter == 0:
			self.G.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'G1.pt'), map_location = self.device))

		self.D = models.Discriminator(self.image_size, self.image_channels * (2 if self.unknown_mode == 'dis' else 1), self.conv_channels, self.conv_layers, self.fc_features, self.dis_cla_fc_layers)
		self.D.to(self.device)
		self.add_model('D', self.D)

		self.optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.paired_dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		self.test_image = torch.load(os.path.join(self.save_path, 'samples', 'test_image.pt'))
		self.test_ucode = torch.load(os.path.join(self.save_path, 'samples', 'test_ucode.pt'))

		self.sample_padding = max(self.image_size // 64, 1) * 2
		t = (self.test_image[:, :, 0].mean() + self.test_image[:, :, -1].mean() + self.test_image[:, :, :, 0].mean() + self.test_image[:, :, :, -1].mean()).item() / 4
		self.border_value = (t + 1) / 2

		if self.start_iter == 0:
			self.visualize()
		self.add_periodic_action(self.visualize, self.sample_interval)

	def generate_random_ucode(self):
		noise = gaussian_noise(self.batch_size, self.unknown_size).to(self.device)
		return noise.mul(self.ucode_stats['eigval'].sqrt().unsqueeze(0)).matmul(self.ucode_stats['eigvec'].t()).mul(self.ucode_stats['std'].unsqueeze(0)) + (self.ucode_stats['mean'].unsqueeze(0))

	def ucode_match_func(self, input, target):
		return (input - target).div(self.ucode_stats['std']).pow(2).sum(1).mean()

	def next_batch(self):
		try:
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			batch = next(self.data_iter)

		image, ucode = batch
		umean, ustd = ucode.to(self.device).unbind(1)

		return image.to(self.device), umean, ustd

	def visualize(self):
		output = []
		with torch.no_grad():
			for ucode in self.test_ucode.split(self.batch_size):
				rec = self.G(ucode[:, 0].to(self.device))[0].cpu()
				output.append(rec)
			output = torch.cat(output, 0).add(1).div(2).clamp(min = 0, max = 1)
		filename = os.path.join(self.save_path, 'samples', 'stage2_{0}.jpg'.format(self.current_iter))
		torchvision.utils.save_image(output, filename, nrow = self.sample_col, padding = self.sample_padding, pad_value = 255 if self.border_value < 0.5 else 0)

	def iter_func(self):
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor

		image, umean, ustd = self.next_batch()

		self.optimizer.zero_grad()

		ucode = sample_gaussian(umean, ustd)

		if self.unknown_mode == 'dis':
			with torch.no_grad():
				rec_fixed = self.G_fixed(ucode)[0]
			dis_real_input = torch.cat((image, rec_fixed), 1)
		else:
			dis_real_input = image

		if self.has_rec_branch or not self.random_ucode:
			rec, rec_std = self.G(ucode)
			rec_t = rec.detach().requires_grad_()

			rec_loss = rec_loss_func(rec_t, image, rec_std)
			rec_grad = autograd.grad(rec_loss * self.rec_weight, rec_t)[0]

			if self.test_rec:
				with torch.no_grad():
					rec_test, rec_test_std = self.G(umean)
					rec_test_loss = rec_loss_func(rec_test, image, rec_test_std)

			if self.unknown_mode == 'dis':
				dis_rec_input = torch.cat((rec_t, rec_fixed), 1)
			else:
				dis_rec_input = rec_t

			dis_real_output, dis_rec_output, dis_op_output = self.D(dis_real_input, dis_rec_input, self.random_offset, self.fake_reflect)
			dis_real_loss = (dis_real_output - 1).pow(2).mean()
			dis_rec_loss = (dis_rec_output + 1).pow(2).mean()
			if self.fake_reflect:
				dis_rec_loss = (dis_rec_loss + (dis_op_output + 1).pow(2).mean()) / 2
			rec_dis_loss = dis_rec_output.pow(2).mean()
			(dis_real_loss + dis_rec_loss * self.branch_factor).backward(retain_graph = True)
			rec_dis_grad = autograd.grad(rec_dis_loss * self.dis_weight * self.branch_factor, rec_t)[0]

			if self.unknown_mode == 'mse':
				rec_ucode = self.E(rec_t)[0][0]
				rec_match_loss = self.ucode_match_func(rec_ucode, umean)
				rec_match_grad = autograd.grad(rec_match_loss * self.match_weight * self.branch_factor, rec_t)[0]
			else:
				rec_match_grad = 0

			rec.backward(rec_grad + rec_dis_grad + rec_match_grad)

		if self.random_ucode:
			ucode_alt = self.generate_random_ucode()
			ucode_alt_target = ucode_alt

			cross = self.G(ucode_alt)[0]
			cross_t = cross.detach().requires_grad_()

			if self.unknown_mode == 'dis':
				with torch.no_grad():
					cross_fixed = self.G_fixed(ucode_alt)[0]
				dis_cross_input = torch.cat((cross_t, cross_fixed), 1)
			else:
				dis_cross_input = cross_t

			dis_cross_output = self.D(dis_cross_input, random_offset = self.random_offset)
			dis_cross_loss = (dis_cross_output + 1).pow(2).mean()
			cross_dis_loss = dis_cross_output.pow(2).mean()
			(dis_cross_loss * self.branch_factor).backward(retain_graph = True)
			cross_dis_grad = autograd.grad(cross_dis_loss * self.dis_weight * self.branch_factor, cross_t)[0]

			if self.unknown_mode == 'mse':
				cross_ucode = self.E(cross_t)[0][0]
				cross_match_loss = self.ucode_match_func(cross_ucode, ucode_alt_target)
				cross_match_grad = autograd.grad(cross_match_loss * self.match_weight * self.branch_factor, cross_t)[0]
			else:
				cross_match_grad = 0

			cross.backward(cross_dis_grad + cross_match_grad)

		self.optimizer.step()

		self.log(dreal = dis_real_loss.item())
		if self.has_rec_branch:
			self.log(rec = rec_loss.item())
			if self.test_rec:
				self.log(rtest = rec_test_loss.item())
			if self.random_ucode:
				self.log(dfake = (dis_rec_loss + dis_cross_loss).item() / 2, gdis = (rec_dis_loss + cross_dis_loss).item() / 2)
				if self.unknown_mode == 'mse':
					self.log(match = (rec_match_loss + cross_match_loss).item() / 2)
			else:
				self.log(dfake = dis_rec_loss.item(), gdis = rec_dis_loss.item())
				if self.unknown_mode == 'mse':
					self.log(match = rec_match_loss.item())
		else:
			self.log(dfake = dis_cross_loss.item(), gdis = cross_dis_loss.item())
			if self.unknown_mode == 'mse':
				self.log(match = cross_match_loss.item())

	def report_loss(self):
		self.print(''.join([
			's2 {iter}:',
			(' r={rec:.3f}' + ('/{rtest:.3f}' if self.test_rec else '')) if self.has_rec_branch else '',
			' m={match:.2f}' if self.unknown_mode == 'mse' else '',
			' d={dreal:.2f}/{dfake:.2f}/{gdis:.2f}'
		]))

	def finalize(self):
		torch.save(self.G.state_dict(), os.path.join(self.save_path, 'trained_models', 'G2.pt'))
