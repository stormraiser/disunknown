import os.path

import torch, torchvision

class QMNIST:

	def __init__(self, root, part, labeled_factors, transform, balanced):
		self.base = torchvision.datasets.QMNIST(root, 'nist', transform = transform)

		prefix = 'balanced_' if balanced else ''
		self.metadata = torch.load(os.path.join(root, prefix + 'metadata.pt'))
		labels = torch.load(os.path.join(root, prefix + ('train' if part == 'train' else 'test') + '.pt'))
		self.image_ids = labels[:, 0]

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
					class_count.append(labels[:, 1].bincount(minlength = 10))
				elif factor == 'writer':
					factor_labels.append(labels[:, 2])
					class_count.append(labels[:, 2].bincount(minlength = self.metadata['num_writers']))
			self.labels = torch.stack(factor_labels, 1)
			self.nclass = [t.size(0) for t in class_count]
			self.class_freq = [t.float() / labels.size(0) for t in class_count]

	def __len__(self):
		return self.image_ids.size(0)

	def __getitem__(self, k):
		image_id = self.image_ids[k]
		image = self.base[image_id][0]

		return image if self.labels is None else (image, self.labels[k])

	@staticmethod
	def add_prepare_args(parser):
		parser.add_argument('--min_balanced_count', type = int, default = 10,
			help = 'minimum number of images per class per writer in balanced training subset')

	@staticmethod
	def prepare_data(args):
		dataset = torchvision.datasets.QMNIST(args.data_path, 'nist', download = True)
		image_ids = torch.arange(len(dataset))
		class_labels = dataset.targets[:, 0]
		writer_labels = dataset.targets[:, 2]


		writer_unique_ids, writer_mapped_labels = torch.unique(writer_labels, return_inverse = True)
		print('{0} writers'.format(writer_unique_ids.size(0)))
		metadata = {
			'num_writers': writer_unique_ids.size(0),
			'orig_writer_ids': writer_unique_ids
		}
		labels = torch.stack((image_ids, class_labels, writer_mapped_labels), 1)
		test_num = int(len(dataset) * args.test_portion + 0.5)

		perm = torch.randperm(len(dataset))
		train_ids = perm[test_num:].sort(0)[0]
		train_labels = labels[train_ids]
		test_ids = perm[:test_num].sort(0)[0]
		test_labels = labels[test_ids]

		torch.save(metadata, os.path.join(args.data_path, 'metadata.pt'))
		torch.save(train_labels, os.path.join(args.data_path, 'train.pt'))
		torch.save(test_labels, os.path.join(args.data_path, 'test.pt'))


		joint_labels = writer_mapped_labels * 10 + class_labels
		joint_count = joint_labels.bincount(minlength = writer_unique_ids.size(0) * 10)
		joint_count_cumsum = torch.cat((torch.tensor([0]), joint_count.cumsum(0)), 0)
		balanced_count = joint_count.reshape(writer_unique_ids.size(0), 10).min(1)[0]
		balanced_count_sort = balanced_count.sort(0, descending = True)[0]
		joint_argsort = (joint_labels.float() + torch.rand(len(dataset)) * 0.1).argsort()
		train_ids = []
		test_ids = []
		for i in range(writer_unique_ids.size(0)):
			if balanced_count[i] >= args.min_balanced_count + 1:
				for j in range(10):
					train_ids.append(joint_argsort[joint_count_cumsum[i * 10 + j] : joint_count_cumsum[i * 10 + j] + balanced_count[i] - 1])
					test_ids.append(joint_argsort[joint_count_cumsum[i * 10 + j] + balanced_count[i] - 1 : joint_count_cumsum[i * 10 + j + 1]])
		train_ids = torch.cat(train_ids, 0).sort(0)[0]
		test_ids = torch.cat(test_ids, 0).sort(0)[0]
		balanced_ids = torch.cat((train_ids, test_ids), 0)
		class_labels = class_labels[balanced_ids]
		writer_labels = writer_labels[balanced_ids]

		writer_unique_ids, writer_mapped_labels = torch.unique(writer_labels, return_inverse = True)
		metadata = {
			'num_writers': writer_unique_ids.size(0),
			'orig_writer_ids': writer_unique_ids
		}
		labels = torch.stack((balanced_ids, class_labels, writer_mapped_labels), 1)
		print('balanced_subset: {0} writers, {1} training, {2} test'.format(writer_unique_ids.size(0), train_ids.size(0), test_ids.size(0)))

		torch.save(metadata, os.path.join(args.data_path, 'balanced_metadata.pt'))
		torch.save(labels[:train_ids.size(0)].clone(), os.path.join(args.data_path, 'balanced_train.pt'))
		torch.save(labels[train_ids.size(0):].clone(), os.path.join(args.data_path, 'balanced_test.pt'))
