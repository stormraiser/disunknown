import torch, torchvision

class MNIST(torchvision.datasets.MNIST):

	def __init__(self, root, part, labeled_factors, transform):
		super().__init__(root, part == 'train', transform = transform, download = True)
		if len(labeled_factors) == 0:
			self.has_label = False
			self.nclass = []
			self.class_freq = []
		else:
			self.has_label = True
			self.nclass = [10]
			class_count = self.targets.bincount(minlength = 10)
			self.class_freq = [class_count.float() / self.data.size(0)]

	def __getitem__(self, k):
		img, target = super().__getitem__(k)
		return (img, torch.tensor([target])) if self.has_label else img

class FashionMNIST(torchvision.datasets.FashionMNIST):

	def __init__(self, root, part, labeled_factors, transform):
		super().__init__(root, part == 'train', transform = transform, download = True)
		if len(labeled_factors) == 0:
			self.has_label = False
			self.nclass = []
			self.class_freq = []
		else:
			self.has_label = True
			self.nclass = [10]
			class_count = self.targets.bincount(minlength = 10)
			self.class_freq = [class_count.float() / self.data.size(0)]

	def __getitem__(self, k):
		img, target = super().__getitem__(k)
		return (img, torch.tensor([target])) if self.has_label else img

class EMNIST(torchvision.datasets.EMNIST):

	def __init__(self, root, part, labeled_factors, transform, split):
		super().__init__(root, split = split, train = part == 'train', transform = transform, download = True)
		if len(labeled_factors) == 0:
			self.has_label = False
			self.nclass = []
			self.class_freq = []
		else:
			self.has_label = True
			class_count = self.targets.bincount()
			self.nclass = [class_count.size(0)]
			self.class_freq = [class_count.float() / self.data.size(0)]

		self.data = self.data.transpose(1, 2)

	def __getitem__(self, k):
		img, target = super().__getitem__(k)
		return (img, torch.tensor([target])) if self.has_label else img