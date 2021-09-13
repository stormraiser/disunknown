import math

import torch

def run_encoder(dataloader, encoder, device):
	output = None
	label = []
	with torch.no_grad():
		for batch in dataloader:
			if isinstance(batch, list):
				batch_image, batch_label = batch
				label.append(batch_label)
			else:
				batch_image = batch
			batch_output = encoder(batch_image.to(device))
			if output is None:
				output = [[torch.stack(t, 1).cpu()] for t in batch_output]
			else:
				for k, t in enumerate(batch_output):
					output[k].append(torch.stack(t, 1).cpu())
	output = [torch.cat(t, 0) for t in output]
	return output if len(label) == 0 else (output, torch.cat(label, 0))

def get_code_stats(code, device):
	code = code.to(torch.float64).to(device)
	n = code.size(0)
	m = code.size(2)
	s = torch.zeros(m, dtype = torch.float64, device = device)
	sxy = torch.zeros(m, m, dtype = torch.float64, device = device)
	sv = torch.zeros(m, dtype = torch.float64, device = device)

	for code_batch in code.split(1024):
		mean, std = code_batch.unbind(1)
		s.add_(mean.sum(0))
		sxy.add_(mean.t() @ mean)
		sv.add_(std.pow(2).sum(0))

	ex = s.cpu() / n
	cov = sxy.cpu() / n - ex.unsqueeze(0) * ex.unsqueeze(1)
	ev = sv.cpu() / n

	full_var = torch.diag(cov) + ev
	full_std = full_var.sqrt()
	var_ratio = torch.diag(cov) / full_var

	normalized_cov = cov / (full_std.unsqueeze(0) * full_std.unsqueeze(1))

	eigval, eigvec = torch.linalg.eigh(normalized_cov)

	stats = {
		'mean': ex.to(torch.float32),
		'std': full_std.to(torch.float32),
		'var_ratio': var_ratio.to(torch.float32),
		'eigval': eigval.to(torch.float32),
		'eigvec': eigvec.to(torch.float32)
	}
	return stats