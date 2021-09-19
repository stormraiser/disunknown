import os.path, io

from PIL import Image

import numpy as np
import torch

class DSprites:

	def __init__(self, root, part, labeled_factors, transform, **kwargs):
		self.root = root
		self.transform = transform
		data = np.load(os.path.join(self.root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
		index = torch.load(os.path.join(root, part + '.pt'))

		labels = torch.tensor(data['latents_classes']).reshape(18, 40, -1)[:, :39].reshape(-1, 6)
		xy_labels = labels[index, 4:]
		self.growth_start = kwargs.get('growth_start', 0)
		self.growth_rate = kwargs.get('growth_rate', 1)
		x_dist = torch.min((xy_labels[:, 0] - 15).abs(), (xy_labels[:, 0] - 16).abs())
		y_dist = torch.min((xy_labels[:, 1] - 15).abs(), (xy_labels[:, 1] - 16).abs())
		center_dist = torch.max(x_dist, y_dist)
		dist_argsort = torch.argsort(center_dist, 0)
		index = index[dist_argsort]
		dist_count = center_dist.bincount(minlength = 16)
		self.len_by_timestep = dist_count.cumsum(0)
		self.current_timestep = 15

		if os.path.exists(os.path.join(root, 'images.npy')):
			self.images = np.load(os.path.join(root, 'images.npy')).reshape(18, 40, -1)[:, :39].reshape(-1)[index]
			self.mode = 'compressed'
		else:
			self.images = data['imgs'].reshape(18, 40, -1)[:, :39].reshape(-1, 64, 64)[index] * 255
			self.mode = 'raw'

		if len(labeled_factors) == 0:
			self.labels = None
			self.nclass = []
			self.class_freq = []
		else:
			factor_labels = []
			self.nclass = []
			for factor in labeled_factors:
				if factor == 'shape':
					factor_labels.append(labels[:, 1])
					self.nclass.append(3)
				elif factor == 'scale':
					factor_labels.append(labels[:, 2])
					self.nclass.append(6)
				elif factor == 'orientation':
					ori = labels[:, 3]
					if kwargs.get('relabel_orientation', False):
						ori = ori.reshape(3, -1)
						square_ori = (ori[0] * 4) % 39
						ellipse_ori = (ori[1] * 2) % 39
						heart_ori = ori[2]
						ori = torch.stack((square_ori, ellipse_ori, heart_ori)).reshape(-1)
					factor_labels.append(ori)
					self.nclass.append(39)
				elif factor == 'x':
					factor_labels.append(labels[:, 4])
					self.nclass.append(32)
				elif factor == 'y':
					factor_labels.append(labels[:, 5])
					self.nclass.append(32)
			self.labels = torch.stack(factor_labels, 1)[index]
			self.class_freq = [self.labels[:, k].bincount(minlength = self.nclass[k]).float() / self.labels.size(0) for k in range(len(labeled_factors))]

	def update(self, current_iter):
		self.current_timestep = min(max(current_iter - self.growth_start, 0) // self.growth_rate, 12) + 3
		print('dataset updated to time step', self.current_timestep)
		n = self.len_by_timestep[self.current_timestep]
		self.class_freq = [self.labels[:n, k].bincount(minlength = self.nclass[k]).float() / n for k in range(len(self.nclass))]

	def __len__(self):
		return self.len_by_timestep[self.current_timestep]

	def __getitem__(self, k):
		if self.mode == 'compressed':
			image = Image.open(io.BytesIO(self.images[k]))
		else:
			image = Image.fromarray(self.images[k])

		if self.transform is not None:
			image = self.transform(image)
		return image if self.labels is None else (image, self.labels[k])

	@staticmethod
	def add_prepare_args(parser):
		parser.add_argument('--compress', action = 'store_true',
			help = 'extract images and compress in PNG format')
		parser.add_argument('--plot_portion', type = float, default = 0.03,
			help = 'portion of samples for plotting')

	@staticmethod
	def prepare_data(args):
		data_num = 3 * 6 * 39 * 32 * 32
		test_num = int(data_num * args.test_portion + 0.5)
		plot_num = min(test_num, int(data_num * args.plot_portion + 0.5))

		perm = torch.randperm(data_num)
		train_ids = perm[test_num:].sort(0)[0]
		test_ids = perm[:test_num].sort(0)[0]
		plot_ids = perm[:plot_num].sort(0)[0]
		torch.save(train_ids, os.path.join(args.data_path, 'train.pt'))
		torch.save(test_ids, os.path.join(args.data_path, 'test.pt'))
		torch.save(plot_ids, os.path.join(args.data_path, 'plot.pt'))

		if args.compress and not os.path.exists(os.path.join(args.data_path, 'images.npy')):
			images = np.load(os.path.join(args.data_path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))['imgs'] * 255
			converted = []
			for k in range(3 * 6 * 40 * 32 * 32):
				image = Image.fromarray(images[k])
				png_bytes = io.BytesIO()
				image.save(png_bytes, format = 'PNG', optimize = True)
				converted.append(png_bytes.getvalue())
			converted = np.array(converted)
			np.save(os.path.join(args.data_path, 'images.npy'), converted)
