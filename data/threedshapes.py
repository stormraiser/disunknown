import os.path, io
import numpy as np

from PIL import Image

import torch

class ThreeDShapes:

	def __init__(self, root, part, labeled_factors, transform):
		self.root = root
		self.transform = transform
		labels = torch.load(os.path.join(root, part + '.pt'))
		self.image_ids = labels[:, 0]
		if os.path.exists(os.path.join(root, 'images.npy')):
			self.images = np.load(os.path.join(root, 'images.npy'))[self.image_ids.numpy()]
			self.mode = 'compressed'
		else:
			import h5py
			h5data = h5py.File(os.path.join(root, '3dshapes.h5'), 'r')
			self.images = h5data['images'][()]
			self.mode = 'raw'

		if len(labeled_factors) == 0:
			self.labels = None
			self.nclass = []
			self.class_freq = []
		else:
			class_count = []
			factor_labels = []
			for factor in labeled_factors:
				if factor == 'floor_hue':
					factor_labels.append(labels[:, 1])
					class_count.append(labels[:, 1].bincount(minlength = 10))
				elif factor == 'wall_hue':
					factor_labels.append(labels[:, 2])
					class_count.append(labels[:, 2].bincount(minlength = 10))
				elif factor == 'object_hue':
					factor_labels.append(labels[:, 3])
					class_count.append(labels[:, 3].bincount(minlength = 10))
				elif factor == 'scale':
					factor_labels.append(labels[:, 4])
					class_count.append(labels[:, 4].bincount(minlength = 8))
				elif factor == 'shape':
					factor_labels.append(labels[:, 5])
					class_count.append(labels[:, 5].bincount(minlength = 4))
				elif factor == 'orientation':
					factor_labels.append(labels[:, 6])
					class_count.append(labels[:, 6].bincount(minlength = 15))
			self.labels = torch.stack(factor_labels, 1)
			self.nclass = [t.size(0) for t in class_count]
			self.class_freq = [t.float() / labels.size(0) for t in class_count]

	def __len__(self):
		return self.image_ids.size(0)

	def __getitem__(self, k):
		if self.mode == 'compressed':
			image = Image.open(io.BytesIO(self.images[k]))
		else:
			image_id = self.image_ids[k]
			image = Image.fromarray(self.images[image_id], mode = 'RGB')

		if self.transform is not None:
			image = self.transform(image)
		return image if self.labels is None else (image, self.labels[k])

	@staticmethod
	def add_prepare_args(parser):
		parser.add_argument('--compress', action = 'store_true',
			help = 'extract images and compress in PNG format')
		parser.add_argument('--plot_portion', type = float, default = 0.05,
			help = 'portion of samples for plotting')

	@staticmethod
	def prepare_data(args):
		import h5py
		h5data = h5py.File(os.path.join(args.data_path, '3dshapes.h5'), 'r')

		test_num = int(480000 * args.test_portion + 0.5)
		plot_num = min(test_num, int(480000 * args.plot_portion + 0.5))
		labels = torch.tensor(h5data['labels'])

		int_labels = [torch.arange(labels.size(0))]
		int_labels.append((labels[:, 0] * 10 + 0.5).floor().to(torch.long))
		int_labels.append((labels[:, 1] * 10 + 0.5).floor().to(torch.long))
		int_labels.append((labels[:, 2] * 10 + 0.5).floor().to(torch.long))
		int_labels.append(((labels[:, 3] - 0.75) * 14 + 0.5).floor().to(torch.long))
		int_labels.append((labels[:, 4] + 0.5).floor().to(torch.long))
		int_labels.append(((labels[:, 5] + 30) / 60 * 14 + 0.5).floor().to(torch.long))
		int_labels = torch.stack(int_labels, 1)

		perm = torch.randperm(labels.size(0))
		train_ids = perm[test_num:].sort(0)[0]
		train_labels = int_labels[train_ids]
		test_ids = perm[:test_num].sort(0)[0]
		test_labels = int_labels[test_ids]
		plot_ids = perm[:plot_num].sort(0)[0]
		plot_labels = int_labels[plot_ids]

		torch.save(train_labels, os.path.join(args.data_path, 'train.pt'))
		torch.save(test_labels, os.path.join(args.data_path, 'test.pt'))
		torch.save(plot_labels, os.path.join(args.data_path, 'plot.pt'))

		if args.compress and not os.path.exists(os.path.join(args.data_path, 'images.npy')):
			images = h5data['images'][()]
			converted = []
			for k in range(480000):
				image = Image.fromarray(images[k], mode = 'RGB')
				png_bytes = io.BytesIO()
				image.save(png_bytes, format = 'PNG', optimize = True)
				converted.append(png_bytes.getvalue())
			converted = np.array(converted)
			np.save(os.path.join(args.data_path, 'images.npy'), converted)