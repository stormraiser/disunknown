import matplotlib
import matplotlib.pyplot as plt

import torch

def plot_one_code(code, stats, dims, color, colormap, filename):
	if dims is None:
		if code.size(2) == 1:
			dims = (0,)
		else:
			t = stats['var_ratio'].ge(0.9).nonzero()
			if t.size(0) < 2:
				t = stats['var_ratio'].argsort(0, descending = True)
			dims = (t[0].item(), t[1].item())
	code0 = code[:, 0, dims[0]].sub(stats['mean'][dims[0]]).div(stats['std'][dims[0]])
	if len(dims) == 1:
		code1 = torch.zeros_like(code0)
	else:
		code1 = code[:, 0, dims[1]].sub(stats['mean'][dims[1]]).div(stats['std'][dims[1]])

	fig, ax = plt.subplots()
	fig.set_size_inches(600 / fig.dpi, 600 / fig.dpi)
	ax.set_aspect(1)
	ax.set_xlim(-3, 3)
	ax.set_ylim(-3, 3)
	ax.scatter(code0, code1, s = 1, c = color, cmap = colormap, vmin = 0, vmax = 1, marker = '.')
	plt.savefig(filename)
	plt.close(fig)

def plot_one_embedding(mean, std, var_ratio, dims, colormap, filename):
	if dims is None:
		if mean.size(1) == 1:
			dims = (0,)
		else:
			t = var_ratio.ge(0.9).nonzero()
			if t.size(0) < 2:
				t = var_ratio.argsort(0, descending = True)
			dims = (t[0].item(), t[1].item())
	mean0 = mean[:, dims[0]]
	std0 = std[:, dims[0]]
	if len(dims) == 1:
		mean1 = torch.zeros_like(mean0)
		std1 = std0
	else:
		mean1 = mean[:, dims[1]]
		std1 = std[:, dims[1]]

	colormap = plt.get_cmap(colormap)

	fig, ax = plt.subplots()
	fig.set_size_inches(600 / fig.dpi, 600 / fig.dpi)
	ax.set_aspect(1)
	ax.set_xlim(-3, 3)
	ax.set_ylim(-3, 3)
	for i in range(mean.size(0)):
		edgecolor = colormap(i / mean.size(0))
		facecolor = edgecolor[:3] + (0.5,)
		ellipse = matplotlib.patches.Ellipse((mean0[i].item(), mean1[i].item()), std0[i].item() * 4, std1[i].item() * 4, ec = edgecolor, fc = facecolor)
		ax.add_patch(ellipse)
	plt.savefig(filename)
	plt.close(fig)
