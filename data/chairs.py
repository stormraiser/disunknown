import os, os.path, io
import numpy as np

from PIL import Image

import torch

altitude = [20, 30]
azimuth = [
	  0,  11,  23,  34,  46,
	 58,  69,  81,  92, 104,
	116, 127, 139, 150, 162,
	174, 185, 197, 209, 220,
	232, 243, 255, 267, 278,
	290, 301, 313, 325, 336,
	348
]

def make_image_name(alt_id, az_id):
	return 'renders/image_{0:03d}_p{1:03d}_t{2:03d}_r096.png'.format(
		alt_id * 32 + az_id,
		altitude[alt_id],
		azimuth[az_id]
	)

def class_from_index(index):
	return index // 62

def altitude_from_index(index):
	return (index % 62) // 31

def azimuth_from_index(index):
	return index % 31

class Chairs:

	def __init__(self, root, part, labeled_factors, transform):
		self.root = root
		self.transform = transform
		self.class_names = sorted([t for t in os.listdir(os.path.join(root, 'rendered_chairs')) if os.path.isdir(os.path.join(root, 'rendered_chairs', t))])
		labels = torch.load(os.path.join(root, ('train' if part == 'train' else 'test') + '.pt'))
		self.image_ids = labels[:, 0]
		if os.path.exists(os.path.join(root, 'images.npy')):
			self.images = np.load(os.path.join(root, 'images.npy'))[self.image_ids.numpy()]
			self.mode = 'numpy'
		else:
			self.mode = 'files'

		if len(labeled_factors) == 0:
			self.labels = None
			self.nclass = []
			self.class_freq = []
		else:
			class_count = []
			factor_labels = []
			for factor in labeled_factors:
				if factor == 'class':
					factor_labels.append(labels[:, 1])
					class_count.append(labels[:, 1].bincount(minlength = len(self.class_names)))
				elif factor == 'altitude':
					factor_labels.append(labels[:, 2])
					class_count.append(labels[:, 2].bincount(minlength = 2))
				elif factor == 'azimuth':
					factor_labels.append(labels[:, 3])
					class_count.append(labels[:, 3].bincount(minlength = 31))
			self.labels = torch.stack(factor_labels, 1).to(torch.long)
			self.nclass = [t.size(0) for t in class_count]
			self.class_freq = [t.float() / labels.size(0) for t in class_count]

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, k):
		if self.mode == 'numpy':
			image = Image.open(io.BytesIO(self.images[k]))
		else:
			image_id = self.image_ids[k].item()
			class_id, alt_id, az_id = class_from_index(image_id), altitude_from_index(image_id), azimuth_from_index(image_id)
			image_path = os.path.join(self.root, 'rendered_chairs', self.class_names[class_id], make_image_name(alt_id, az_id))
			image = Image.open(image_path)

		if self.transform is not None:
			image = self.transform(image)
		return image if self.labels is None else (image, self.labels[k])

	@staticmethod
	def add_prepare_args(parser):
		parser.add_argument('--compress', action = 'store_true',
			help = 'downsample and save whole dataset as npy')
		parser.add_argument('--downsample_size', type = int, default = 128,
			help = 'size of downsampled images')

	@staticmethod
	def prepare_data(args):
		class_names = sorted([t for t in os.listdir(os.path.join(args.data_path, 'rendered_chairs')) if os.path.isdir(os.path.join(args.data_path, 'rendered_chairs', t))])
		data_num = len(class_names) * 62
		test_num = int(data_num * args.test_portion + 0.5)

		image_ids = torch.arange(data_num)
		class_ids = image_ids.div(62, rounding_mode = 'floor')
		altitude_ids = (image_ids % 62).div(31, rounding_mode = 'floor')
		azimuth_ids = image_ids % 31
		index = torch.stack((image_ids, class_ids, altitude_ids, azimuth_ids), 1)

		perm = torch.randperm(data_num)
		train_ids = perm[test_num:].sort(0)[0]
		test_ids = perm[:test_num].sort(0)[0]
		torch.save(index[train_ids], os.path.join(args.data_path, 'train.pt'))
		torch.save(index[test_ids], os.path.join(args.data_path, 'test.pt'))

		if args.compress and not os.path.exists(os.path.join(args.data_path, 'images.npy')):
			converted = []
			for k in range(data_num):
				class_id, alt_id, az_id = class_from_index(k), altitude_from_index(k), azimuth_from_index(k)
				image_path = os.path.join(args.data_path, 'rendered_chairs', class_names[class_id], make_image_name(alt_id, az_id))
				image = Image.open(image_path)
				image = image.resize((args.downsample_size, args.downsample_size))

				png_bytes = io.BytesIO()
				image.save(png_bytes, format = 'PNG', optimize = True)
				converted.append(png_bytes.getvalue())
			converted = np.array(converted)
			np.save(os.path.join(args.data_path, 'images.npy'), converted)