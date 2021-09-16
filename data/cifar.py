import torch, torchvision

class CIFAR10(torchvision.datasets.CIFAR10):

	def __init__(self, root, part, labeled_factors, transform):
		super().__init__(root, part == 'train', transform = transform, download = True)

		if len(labeled_factors) == 0:
			self.has_label = False
			self.nclass = []
			self.class_freq = []
		else:
			self.has_label = True
			self.nclass = [10]
			class_count = torch.tensor(self.targets).bincount(minlength = 10)
			self.class_freq = [class_count.float() / self.data.shape[0]]

	def __getitem__(self, k):
		img, target = super().__getitem__(k)
		return (img, torch.tensor([target])) if self.has_label else img
