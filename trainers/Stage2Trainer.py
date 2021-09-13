import os.path

import torch, torchvision
from torch import optim, autograd
import torch.nn.functional as F

import data, models
from .StageTrainer import StageTrainer
from .utils import *
from .encode_tools import *

class Stage2Trainer(StageTrainer):

	def __init__(self, config):
		super().__init__(config)

		ucode_dict = torch.load(os.path.join(self.save_path, 'trained_models', 'unknown_code.pt'))
		self.ucode_dataset = torch.utils.data.TensorDataset(ucode_dict['train'])
		self.ucode_stats = ucode_dict['train_stats']
		for key in self.ucode_stats:
			self.ucode_stats[key] = self.ucode_stats[key].to(self.device)
		self.paired_dataset = data.PairedDataset(self.dataset, self.ucode_dataset)
		
		self.branch_factor = 0.5 if self.has_rec_branch else 1

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
			self.B_fixed = models.MultiLabelEmbedder(self.nclass, self.labeled_size, self.class_freq)
			self.B_fixed.to(self.device)
			self.B_fixed.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'B1.pt'), map_location = self.device))

		self.G = models.Generator(*model_config, self.enc_gen_fc_layers, [self.unknown_size] + self.labeled_size)
		self.G.to(self.device)
		self.add_model('G', self.G)
		if self.start_iter == 0:
			self.G.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'G1.pt'), map_location = self.device))

		self.D = models.Discriminator(self.image_size, self.image_channels * (2 if self.unknown_mode == 'dis' else 1), self.conv_channels, self.conv_layers, self.fc_features, self.dis_cla_fc_layers)
		self.D.to(self.device)
		self.add_model('D', self.D)

		param_groups = [{'params': self.G.parameters()}, {'params': self.D.parameters()}]

		if self.use_embedding:
			self.B = models.MultiLabelEmbedder(self.nclass, self.labeled_size, self.class_freq)
			self.B.to(self.device)
			self.add_model('B', self.B)
			param_groups.append({'params': self.B.parameters(), 'lr': self.emb_lr})
			if self.start_iter == 0:
				self.B.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'B1.pt'), map_location = self.device))
		else:
			self.S = models.Encoder(*model_config, self.enc_gen_fc_layers, self.labeled_size, zero_init = True)
			self.S.to(self.device)
			self.add_model('S', self.S)
			param_groups.append({'params': self.S.parameters()})

		self.R = models.Stage2Classifier(*model_config, self.dis_cla_fc_layers, self.nclass)
		self.R.to(self.device)
		if self.cla2_adv:
			self.add_model('R', self.R)
			param_groups.append({'params': self.R.parameters()})
		if self.start_iter == 0 or not self.cla2_adv:
			self.R.load_state_dict(torch.load(os.path.join(self.save_path, 'trained_models', 'R1.pt'), map_location = self.device))

		self.optimizer = optim.Adam(param_groups, lr = self.lr, eps = 1e-8)
		self.add_model('optimizer', self.optimizer)

		self.data_loader = torch.utils.data.DataLoader(self.paired_dataset, batch_size = self.batch_size, shuffle = True, drop_last = True, num_workers = self.num_workers)
		self.data_iter = iter(self.data_loader)

		if self.nested_dropout:
			self.lprob = [torch.tensor(t, device = self.device) for t in self.labeled_keep_prob]
		else:
			self.lprob = [torch.ones(t, device = self.device) for t in self.labeled_size]

		self.test_image = torch.load(os.path.join(self.save_path, 'samples', 'test_image.pt'))
		self.test_label = torch.load(os.path.join(self.save_path, 'samples', 'test_label.pt'))
		self.test_ucode = torch.load(os.path.join(self.save_path, 'samples', 'test_ucode.pt'))

		self.sample_padding = max(self.image_size // 64, 1) * 2
		t = (self.test_image[:, :, 0].mean() + self.test_image[:, :, -1].mean() + self.test_image[:, :, :, 0].mean() + self.test_image[:, :, :, -1].mean()).item() / 4
		self.border_value = (t + 1) / 2

		if self.start_iter == 0:
			self.visualize()
		self.add_periodic_action(self.visualize, self.sample_interval)

		if len(self.plot_config) > 0:
			from .plot_tools import plot_one_code, plot_one_embedding

			if self.use_embedding:
				for config in self.plot_config:
					if 'embedding_factor' in config:
						self.plot_one_embedding = plot_one_embedding
						self.add_periodic_action(self.plot_embedding, self.plot_embedding_interval)
						if self.start_iter == 0:
							self.plot_embedding()
						break
			else:
				for config in self.plot_config:
					if 'code_factor' in config and config['code_factor'] != 'unknown':
						self.plot_dataset = self.dataset_class(self.data_path, 'plot', self.all_factors, self.transform, **self.dataset_args)
						self.plot_one_code = plot_one_code
						self.add_periodic_action(self.plot_code, self.plot_code_interval)
						if self.start_iter == 0:
							self.plot_code()
						break

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

		image, label, ucode = batch
		umean, ustd = ucode.to(self.device).unbind(1)

		return image.to(self.device), label.to(self.device), umean, ustd

	def visualize(self):
		output = []
		output_alt = []
		with torch.no_grad():
			if self.use_embedding:
				test_lcode = [t[0] for t in self.B(self.test_label.to(self.device))]
			else:
				test_lcode = [[] for i in range(len(self.labeled_factors))]
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
		filename = os.path.join(self.save_path, 'samples', 'stage2_{0}.jpg'.format(self.current_iter))
		self.save_image_pair(output, output_alt, filename, self.sample_padding, self.border_value)

	def iter_func(self):
		lr_factor = min(self.current_iter / self.lr_ramp, 1)
		self.optimizer.param_groups[0]['lr'] = self.lr * lr_factor
		self.optimizer.param_groups[1]['lr'] = self.lr * lr_factor
		if self.use_embedding:
			self.optimizer.param_groups[2]['lr'] = self.emb_lr * lr_factor
		else:
			self.optimizer.param_groups[2]['lr'] = self.lr * lr_factor
		if self.cla2_adv:
			self.optimizer.param_groups[3]['lr'] = self.lr * lr_factor

		image, label, umean, ustd = self.next_batch()
		offset = torch.randint(1, self.batch_size, (self.batch_size,))
		label_alt = torch.stack([torch.cat((t[k:], t[:k]), 0) for t, k in zip(label.t(), offset)], 1)

		self.optimizer.zero_grad()

		ucode = sample_gaussian(umean, ustd)

		if self.use_embedding:
			lparam = self.B(label)
			ret = self.B.code_loss()
			lcode_loss = sum([t.mul(p).sum() for t, p in zip(ret, self.lprob)])
			lcode_loss_log = sum([t.sum().item() for t in ret])
			lbatch_loss = 0
		else:
			lparam = self.S(image)
			ret = [code_loss_func(*t, p) for t, p in zip(lparam, self.lprob)]
			lcode_loss = sum([t[0] for t in ret])
			lcode_loss_log = sum([t[1] for t in ret])
			lbatch_loss = sum([t[2] for t in ret])
			lbatch_loss_log = sum([t[3] for t in ret])

		ret = [drop_and_sample(*t, p) for t, p in zip(lparam, self.lprob)]
		lmask = [t[0] for t in ret]
		lcode = [t[1] for t in ret]
		lcode_t = [t.detach().requires_grad_() for t in lcode]

		if self.unknown_mode == 'dis':
			with torch.no_grad():
				lcode_fixed = [t[0] for t in self.B_fixed(label)]
				rec_fixed = self.G_fixed(ucode, *lcode_fixed)[0]
			dis_real_input = torch.cat((image, rec_fixed), 1)
		else:
			dis_real_input = image

		if self.has_rec_branch:
			rec, rec_std = self.G(ucode, *lcode_t, None, *lmask)
			rec_t = rec.detach().requires_grad_()

			rec_loss = rec_loss_func(rec_t, image, rec_std)
			rec_grad = autograd.grad(rec_loss * self.rec_weight, rec_t)[0]

			if self.test_rec:
				with torch.no_grad():
					rec_test, rec_test_std = self.G(umean, *[t[0] for t in lparam])
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

			if self.cla2_adv:
				cla2_real_output, cla2_rec_output, cla2_op_output = self.R(image, rec_t, self.random_offset, self.fake_reflect)
				cla2_real_loss = sum([F.nll_loss(c, l) for c, l in zip(cla2_real_output, label.t())])
				cla2_rec_loss = sum([-(1 - torch.gather(c, 1, l.unsqueeze(1)).squeeze(1).exp()).clamp(min = 1e-8).log().mean() for c, l in zip(cla2_rec_output, label.t())])
				if self.fake_reflect:
					cla2_op_loss = sum([-(1 - torch.gather(c, 1, l.unsqueeze(1)).squeeze(1).exp()).clamp(min = 1e-8).log().mean() for c, l in zip(cla2_op_output, label.t())])
					cla2_rec_loss = (cla2_rec_loss + cla2_op_loss) / 2
				(cla2_real_loss + cla2_rec_loss * self.branch_factor).backward(retain_graph = True)
			else:
				cla2_rec_output = self.R(rec_t, random_offset = self.random_offset, dummy_fake = True)
			rec_cla2_loss = sum([F.nll_loss(c, l) for c, l in zip(cla2_rec_output, label.t())])
			rec_cla2_grad = autograd.grad(rec_cla2_loss * self.cla2_weight * self.branch_factor, rec_t)[0]

			if self.unknown_mode == 'mse':
				rec_ucode = self.E(rec_t)[0][0]
				rec_match_loss = self.ucode_match_func(rec_ucode, umean)
				rec_match_grad = autograd.grad(rec_match_loss * self.match_weight * self.branch_factor, rec_t)[0]
			else:
				rec_match_grad = 0

			rec.backward(rec_grad + rec_dis_grad + rec_cla2_grad + rec_match_grad)

		else:
			dis_real_output = self.D(dis_real_input, random_offset = self.random_offset)
			dis_real_loss = (dis_real_output - 1).pow(2).mean()
			dis_real_loss.backward()

			if self.cla2_adv:
				cla2_real_output = self.R(image, random_offset = self.random_offset, dummy_fake = True)
				cla2_real_loss = sum([F.nll_loss(c, l) for c, l in zip(cla2_real_output, label.t())])
				cla2_real_loss.backward()

		lcode_t_alt = [torch.cat((t[k:], t[:k]), 0) for t, k in zip(lcode_t, offset)]
		lmask_alt = [torch.cat((t[k:], t[:k]), 0) for t, k in zip(lmask, offset)]
		if self.random_ucode:
			ucode_alt = self.generate_random_ucode()
			ucode_alt_target = ucode_alt
		else:
			ucode_alt = ucode
			ucode_alt_target = umean

		cross = self.G(ucode_alt, *lcode_t_alt, None, *lmask_alt)[0]
		cross_t = cross.detach().requires_grad_()

		if self.unknown_mode == 'dis':
			with torch.no_grad():
				lcode_fixed_alt = [torch.cat((t[k:], t[:k]), 0) for t, k in zip(lcode_fixed, offset)]
				cross_fixed = self.G_fixed(ucode_alt, *lcode_fixed_alt)[0]
			dis_cross_input = torch.cat((cross_t, cross_fixed), 1)
		else:
			dis_cross_input = cross_t

		dis_cross_output = self.D(dis_cross_input, random_offset = self.random_offset)
		dis_cross_loss = (dis_cross_output + 1).pow(2).mean()
		cross_dis_loss = dis_cross_output.pow(2).mean()
		(dis_cross_loss * self.branch_factor).backward(retain_graph = True)
		cross_dis_grad = autograd.grad(cross_dis_loss * self.dis_weight * self.branch_factor, cross_t)[0]

		cla2_cross_output = self.R(cross_t, random_offset = self.random_offset, dummy_fake = True)
		if self.cla2_adv:
			cla2_cross_loss = sum([-(1 - torch.gather(c, 1, l.unsqueeze(1)).squeeze(1).exp()).clamp(min = 1e-8).log().mean() for c, l in zip(cla2_cross_output, label_alt.t())])
			(cla2_cross_loss * self.branch_factor).backward(retain_graph = True)
		cross_cla2_loss = sum([F.nll_loss(c, l) for c, l in zip(cla2_cross_output, label_alt.t())])
		cross_cla2_grad = autograd.grad(cross_cla2_loss * self.cla2_weight * self.branch_factor, cross_t)[0]

		if self.unknown_mode == 'mse':
			cross_ucode = self.E(cross_t)[0][0]
			cross_match_loss = self.ucode_match_func(cross_ucode, ucode_alt_target)
			cross_match_grad = autograd.grad(cross_match_loss * self.match_weight * self.branch_factor, cross_t)[0]
		else:
			cross_match_grad = 0

		cross.backward(cross_dis_grad + cross_cla2_grad + cross_match_grad)

		lcode_grad = [t.grad for t in lcode_t]

		autograd.backward(lcode + [lcode_loss * self.lcode_weight + lbatch_loss * self.lbatch_weight], lcode_grad + [None])

		self.optimizer.step()

		self.log(lcode = lcode_loss_log, dreal = dis_real_loss.item())
		if not self.use_embedding:
			self.log(lbatch = lbatch_loss_log)
		if self.cla2_adv:
			self.log(creal = cla2_real_loss.item())
		if self.has_rec_branch:
			self.log(rec = rec_loss.item(), dfake = (dis_rec_loss + dis_cross_loss).item() / 2, gdis = (rec_dis_loss + cross_dis_loss).item() / 2, gcla = (rec_cla2_loss + cross_cla2_loss).item() / 2)
			if self.test_rec:
				self.log(rtest = rec_test_loss.item())
			if self.cla2_adv:
				self.log(cfake = (cla2_rec_loss + cla2_cross_loss).item() / 2)
			if self.unknown_mode == 'mse':
				self.log(match = (rec_match_loss + cross_match_loss).item() / 2)
		else:
			self.log(dfake = dis_cross_loss.item(), gdis = cross_dis_loss.item(), gcla = cross_cla2_loss.item())
			if self.cla2_adv:
				self.log(cfake = cla2_cross_loss.item())
			if self.unknown_mode == 'mse':
				self.log(match = cross_match_loss.item())

	def report_loss(self):
		self.print(''.join([
			's2 {iter}:',
			(' r={rec:.3f}' + ('/{rtest:.3f}' if self.test_rec else '')) if self.has_rec_branch else '',
			' l={lcode:.2f}',
			'' if self.use_embedding else '/{lbatch:.2f}',
			' m={match:.2f}' if self.unknown_mode == 'mse' else '',
			' d={dreal:.2f}/{dfake:.2f}/{gdis:.2f}',
			' c={creal:.2f}/{cfake:.2f}/{gcla:.2f}' if self.cla2_adv else ' c={gcla:.2f}'
		]))

	def finalize(self):
		torch.save(self.G.state_dict(), os.path.join(self.save_path, 'trained_models', 'G2.pt'))
		if self.use_embedding:
			torch.save(self.B.state_dict(), os.path.join(self.save_path, 'trained_models', 'B2.pt'))
		else:
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
				filename = os.path.join(self.save_path, 'plots', '{0}_{1}_{2}_s2_{3}.png'.format(k, config['code_factor'], config['color_factor'], self.current_iter))
				self.plot_one_code(lcode_plot[code_factor_id], lcode_plot_stats[code_factor_id], config.get('dims'), color, config.get('colormap'), filename)

	def plot_embedding(self):
		lparam = self.B.get_all()
		var_ratio = self.B.get_var_ratio()

		for k, config in enumerate(self.plot_config):
			if 'embedding_factor' in config:
				code_factor_id = self.labeled_factors.index(config['embedding_factor'])
				filename = os.path.join(self.save_path, 'plots', '{0}_{1}_s2_{2}.png'.format(k, config['embedding_factor'], self.current_iter))
				self.plot_one_embedding(*lparam[code_factor_id], var_ratio[code_factor_id], config.get('dims'), config.get('colormap'), filename)
