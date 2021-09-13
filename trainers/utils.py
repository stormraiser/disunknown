import math

import torch

def gaussian_noise(m, k):
    base = torch.stack([torch.randperm(m, dtype = torch.float32) for i in range(k)], 1)
    rand = (base + torch.rand(m, k)) / m * 1.99998 - 0.99999
    return torch.erfinv(rand) * (2 ** 0.5)

def sample_gaussian(mean, std):
    return mean + std.mul(gaussian_noise(std.size(0), std.size(1)).to(std))

def cla1_adv_nlu(input, target, class_freq):
	target_freq = class_freq[target]
	weight = target_freq.clamp(min = 1e-8).reciprocal() - 1
	return -(1 - torch.gather(input, 1, target.unsqueeze(1)).squeeze(1).exp()).clamp(min = 1e-8).log().mul(weight).mean()

def cla1_adv_ll(input, target, class_freq):
	return torch.gather(input, 1, target.unsqueeze(1)).mean()

def cla1_adv_ll_clamp(input, target, class_freq):
	target_freq = class_freq[target]
	limit = target_freq.clamp(min = 1e-8).log()
	return torch.max(torch.gather(input, 1, target.unsqueeze(1)).squeeze(1), limit).mean()

def cla1_adv_ce(input, target, class_freq):
	normalizer = class_freq.pow(2).sum().sqrt()
	return -input.mean(0) @ class_freq / normalizer

def rec_loss_func(input, target, std):
	return ((input - target).pow(2).mean(3).mean(2).mean(1) / std.pow(2) / 2 + std.log() + math.log(math.pi * 2) / 2).mean()

def code_loss_func(mean, std, keep_prob, batch_loss = True):
	code_loss = ((mean.pow(2) + std.pow(2)) * 0.5 - std.log() - 0.5)
	code_loss_log = code_loss.sum(1).mean().item()
	code_loss = code_loss.mean(0).mul(keep_prob).sum()

	if not batch_loss:
		return code_loss, code_loss_log

	batch_mean = mean.mean(0)
	batch_var = (mean.pow(2).mean(0) - batch_mean.pow(2)) * (mean.size(0) / (mean.size(0) - 1)) + std.pow(2).mean(0)
	batch_code_loss = (batch_mean.pow(2) + batch_var - batch_var.log() - 1) * 0.5
	batch_code_loss_log = batch_code_loss.sum().item()
	batch_code_loss = batch_code_loss.mul(keep_prob).sum()

	return code_loss, code_loss_log, batch_code_loss, batch_code_loss_log

def drop_and_sample(mean, std, keep_prob):
	if keep_prob.dim() == 1:
		keep_prob = keep_prob.unsqueeze(0)
	drop_level = torch.rand(mean.size(0), device = mean.device)
	drop_mask = keep_prob.ge(drop_level.unsqueeze(1))
	masked_mean = torch.where(drop_mask, mean, torch.zeros_like(mean))
	masked_std = torch.where(drop_mask, std, torch.ones_like(std))
	return drop_mask.float(), sample_gaussian(masked_mean, masked_std)

def normal_kld(q_mean, q_std, p_mean, p_std):
	return ((p_std.pow(2) + (p_mean - q_mean).pow(2)).div(q_std.pow(2)) * 0.5 - 0.5 + q_std.log() - p_std.log()).sum(1).mean()