import os.path

import torch, torchvision
from torch import optim, autograd
import torch.nn.functional as F

import time

import models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class Stage1Trainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		model_config = (self.image_size, self.image_channels, self.conv_channels, self.conv_layers, self.fc_features)
		code_cla1_config = (self.unknown_size, self.code_cla1_features, self.code_cla1_layers)

		self.E = models.Encoder(*model_config, self.enc_gen_fc_layers, [self.unknown_size], zero_init = self.zero_init)
		self.E.to(self.device)
		self.add_model('E', self.E)

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size] + self.labeled_size)
		self.G.to(self.device)
		self.add_model('G', self.G)

		if self.cla1_mode == 'compound':
			self.C = models.Stage1CompoundClassifier(*model_config, self.dis_cla_fc_layers, *code_cla1_config, self.nclass)
		elif self.cla1_mode == 'image':
			self.C = models.Stage1ImageClassifier(*model_config, self.dis_cla_fc_layers, self.nclass)
		elif self.cla1_mode == 'code':
			self.C = models.Stage1CodeClassifier(*code_cla1_config, self.nclass)
		self.C.to(self.device)
		self.add_model('C', self.C)

		self.B = models.MultiLabelEmbedder(self.nclass, self.labeled_size, self.class_freq, self.labeled_init)
		self.B.to(self.device)
		self.add_model('B', self.B)

		self.optimizer = optim.Adam([
			{'params': self.E.parameters()},
			{'params': self.G.parameters()},
			{'params': self.C.parameters()},
			{'params': self.B.parameters(), 'lr': self.emb_lr}
		], lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.cla1_adv_func = eval('cla1_adv_' + self.cla1_adv_mode)

		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		if self.nested_dropout:
			self.uprob = torch.tensor(self.unknown_keep_prob, device = self.device)
			self.lprob = [torch.tensor(t, device = self.device) for t in self.labeled_keep_prob]
		else:
			self.uprob = torch.ones(self.unknown_size, device = self.device)
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
			self.save_image_pair((self.test_image + 1) / 2, torch.full_like(self.test_image, self.border_value), os.path.join(self.save_path, 'samples', 'test_image.jpg'), self.sample_padding, self.border_value)
			self.visualize()

		self.add_periodic_action(self.visualize, self.sample_interval)

		if len(self.plot_config) > 0:
			from .plot_tools import plot_one_code, plot_one_embedding

			for config in self.plot_config:
				if config.get('code_factor') == 'unknown':
					self.plot_dataset = self.dataset_class(self.data_path, 'plot', self.all_factors, self.transform, **self.dataset_args)
					self.plot_one_code = plot_one_code
					self.add_periodic_action(self.plot_code, self.plot_code_interval)
					if self.start_iter == 0:
						self.plot_code()
					break

			for config in self.plot_config:
				if 'embedding_factor' in config:
					self.plot_one_embedding = plot_one_embedding
					self.add_periodic_action(self.plot_embedding, self.plot_embedding_interval)
					if self.start_iter == 0:
						self.plot_embedding()
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
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			batch = next(self.data_iter)

		image, label = batch

		return image.to(self.device), label.to(self.device)

	def visualize(self):
		output = []
		output_alt = []
		with torch.no_grad():
			test_lcode = [t[0] for t in self.B(self.test_label.to(self.device))]
			test_lcode_alt = [torch.cat((t[1:], t[:1]), 0) for t in test_lcode]

			for i in range((self.test_image.size(0) - 1) // self.batch_size + 1):
				image = self.test_image[i * self.batch_size : (i + 1) * self.batch_size].to(self.device)
				ucode = self.E(image)[0][0]
				lcode = [t[i * self.batch_size : (i + 1) * self.batch_size] for t in test_lcode]
				lcode_alt = [t[i * self.batch_size : (i + 1) * self.batch_size] for t in test_lcode_alt]
				rec = self.G(ucode, *lcode)[0].cpu()
				cross = self.G(ucode, *lcode_alt)[0].cpu()
				output.append(rec)
				output_alt.append(cross)
			output = torch.cat(output, 0).add(1).div(2).clamp(min = 0, max = 1)
			output_alt = torch.cat(output_alt, 0).add(1).div(2).clamp(min = 0, max = 1)
		filename = os.path.join(self.save_path, 'samples', 'stage1_{0}.jpg'.format(self.current_iter))
		self.save_image_pair(output, output_alt, filename, self.sample_padding, self.border_value)

	def iter_func(self):
		lr_factor = max(min((self.current_iter - self.enc_freeze) / self.lr_ramp, 1), 0)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor
		lr_factor = max(min((self.current_iter - self.emb_freeze) / self.lr_ramp, 1), 0)
		self.optimizer.param_groups[3]['lr'] = self.emb_lr * lr_factor
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[1]['lr'] = self.lr * lr_factor
		self.optimizer.param_groups[2]['lr'] = self.lr * lr_factor

		if self.cla1_ramp[1] > 0:
			h = min(max((self.current_iter - self.cla1_ramp[0]) / (self.cla1_ramp[1] - self.cla1_ramp[0]), 0), 1)
		else:
			h = 1
		cla1_weight = h * self.cla1_weight

		image, label = self.next_batch()
		offset = torch.randint(1, self.batch_size, (self.batch_size,))
		label_alt = torch.stack([torch.cat((t[k:], t[:k]), 0) for t, k in zip(label.t(), offset)], 1)

		self.optimizer.zero_grad()

		umean, ustd = self.E(image)[0]
		umask, ucode = drop_and_sample(umean, ustd, self.uprob)
		ucode_loss, ucode_loss_log, ubatch_loss, ubatch_loss_log = code_loss_func(umean, ustd, self.uprob)

		ucode_t = ucode.detach().requires_grad_()

		lparam = self.B(label)
		ret = [drop_and_sample(*t, p) for t, p in zip(lparam, self.lprob)]
		lmask = [t[0] for t in ret]
		lcode = [t[1] for t in ret]
		ret = self.B.code_loss()
		lcode_loss = sum([t.mul(p).sum() for t, p in zip(ret, self.lprob)])
		lcode_loss_log = sum([t.sum().item() for t in ret])

		rec, rec_std = self.G(ucode_t, *lcode, umask, *lmask)
		rec_loss = rec_loss_func(rec, image, rec_std)
		(rec_loss * self.rec_weight).backward()

		if self.test_rec:
			with torch.no_grad():
				rec_test, rec_test_std = self.G(umean, *[t[0] for t in lparam])
				rec_test_loss = rec_loss_func(rec_test, image, rec_test_std)

		if self.cla1_mode == 'code':
			ucode_t2 = ucode.detach().requires_grad_()
			cla1_output = self.C(ucode_t2, umask)
		else:
			lparam_alt = self.B(label_alt)
			ret = [drop_and_sample(*t, p) for t, p in zip(lparam_alt, self.lprob)]
			lmask_alt = [t[0] for t in ret]
			lcode_alt = [t[1] for t in ret]

			for param in self.G.conv_params():
				param.requires_grad_(self.gen_conv_adv)
			for param in self.G.fc_params():
				param.requires_grad_(self.gen_fc_adv)

			cross = self.G(ucode_t, *lcode_alt, umask, *lmask_alt)[0]
			cross_t = cross.detach().requires_grad_()

			if self.cla1_mode == 'compound':
				ucode_t2 = ucode.detach().requires_grad_()
				cla1_output = self.C(cross_t, ucode_t2, umask)
			else:
				cla1_output = self.C(cross_t)

		cla1_loss = sum([F.nll_loss(c, l) for c, l in zip(cla1_output, label.t())])
		cla1_loss.backward(retain_graph = True)
		cross_cla1_loss = sum([self.cla1_adv_func(c, l, f) for c, l, f in zip(cla1_output, label.t(), self.class_freq)])

		if self.cla1_mode == 'code':
			ucode_grad2 = autograd.grad(cross_cla1_loss * cla1_weight, [ucode_t2])[0]
		else:
			if self.cla1_mode == 'compound':
				cross_grad, ucode_grad2 = autograd.grad(cross_cla1_loss * cla1_weight, [cross_t, ucode_t2])
			else:
				cross_grad = autograd.grad(cross_cla1_loss * cla1_weight, [cross_t])[0]
				ucode_grad2 = 0
			cross.backward(cross_grad)

			for param in self.G.conv_params():
				param.requires_grad_(True)
			for param in self.G.fc_params():
				param.requires_grad_(True)

		ucode_grad = ucode_t.grad + ucode_grad2
		autograd.backward([ucode, ucode_loss * self.ucode_weight + ubatch_loss * self.ubatch_weight + lcode_loss * self.lcode_weight], [ucode_grad, None])

		self.optimizer.step()

		self.log(rec = rec_loss.item(), ucode = ucode_loss_log, ubatch = ubatch_loss_log, lcode = lcode_loss_log, cla1 = cla1_loss.item())
		if self.test_rec:
			self.log(rtest = rec_test_loss.item())

	def report_loss(self):
		self.print('s1 {iter}: rec={rec:.3f}' + ('/{rtest:.3f}' if self.test_rec else '') + ' ucode={ucode:.2f}/{ubatch:.2f} lcode={lcode:.2f} cla1={cla1:.2f}')

	def finalize(self):
		torch.save(self.E.state_dict(), os.path.join(self.save_path, 'trained_models', 'E.pt'))
		torch.save(self.G.state_dict(), os.path.join(self.save_path, 'trained_models', 'G1.pt'))
		torch.save(self.B.state_dict(), os.path.join(self.save_path, 'trained_models', 'B1.pt'))

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

	def plot_embedding(self):
		lparam = self.B.get_all()
		var_ratio = self.B.get_var_ratio()

		for k, config in enumerate(self.plot_config):
			if 'embedding_factor' in config:
				code_factor_id = self.labeled_factors.index(config['embedding_factor'])
				filename = os.path.join(self.save_path, 'plots', '{0}_{1}_s1_{2}.png'.format(k, config['embedding_factor'], self.current_iter))
				self.plot_one_embedding(*lparam[code_factor_id], var_ratio[code_factor_id], config.get('dims'), config.get('colormap'), filename)