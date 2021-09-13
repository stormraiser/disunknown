class PairedDataset:

	def __init__(self, dataset1, dataset2):
		self.dataset1 = dataset1
		self.dataset2 = dataset2

	def __len__(self):
		return len(self.dataset1)

	def __getitem__(self, k):
		ret1 = self.dataset1[k]
		ret1 = ret1 if isinstance(ret1, tuple) else (ret1,)
		ret2 = self.dataset2[k]
		ret2 = ret2 if isinstance(ret2, tuple) else (ret2,)
		return ret1 + ret2
