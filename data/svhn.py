import numpy as np
import torch, torchvision

class SVHN:

	def __init__(self, root, part, labeled_factors, transform, include_extra):
		if part == 'train':
			if include_extra:
				train_set = torchvision.datasets.SVHN(root, 'train', transform, download = True)
				extra_set = torchvision.datasets.SVHN(root, 'extra', transform, download = True)
				class_count = np.bincount(train_set.labels) + np.bincount(extra_set.labels)
				self.base = torch.utils.data.ConcatDataset([train_set, extra_set])
			else:
				self.base = torchvision.datasets.SVHN(root, 'train', transform, download = True)
				class_count = np.bincount(self.base.labels)
		else:
			self.base = torchvision.datasets.SVHN(root, 'test', transform, download = True)
			class_count = np.bincount(self.base.labels)

		if len(labeled_factors) == 0:
			self.has_label = False
			self.nclass = []
			self.class_freq = []
		else:
			self.has_label = True
			self.nclass = [10]
			class_count = torch.tensor(class_count)
			self.class_freq = [class_count.float() / class_count.sum()]

	def __len__(self):
		return len(self.base)

	def __getitem__(self, k):
		img, target = self.base[k]
		return (img, torch.tensor([target])) if self.has_label else img
