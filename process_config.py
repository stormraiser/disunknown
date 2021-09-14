import os.path, shutil

import yaml

_default_config_path = os.path.join(os.path.dirname(__file__), 'default_config')

def _make_keep_prob(size, param):
	if param is None:
		return [1] * size
	else:
		param = [1] + param
		if size <= len(param):
			return param[:size]
		else:
			a = (param[-1] / param[-2]) ** (1 / (size - len(param) + 1))
			return param[:-1] + [param[-2] * a ** (k + 1) for k in range(size - len(param) + 1)]

def process_config(args):
	if args.config_file is not None:
		with open(args.config_file) as file:
			raw_config = yaml.load(file, Loader = yaml.Loader)
		os.makedirs(args.save_path, exist_ok = True)
		shutil.copy(args.config_file, os.path.join(args.save_path, 'config.yaml'))
	else:
		with open(os.path.join(args.save_path, 'config.yaml')) as file:
			raw_config = yaml.load(file, Loader = yaml.Loader)
	with open(os.path.join(_default_config_path, 'default_config.yaml')) as file:
		defaults = yaml.load(file, Loader = yaml.Loader)
	dataset_default_path = os.path.join(_default_config_path, 'datasets', raw_config['dataset'] + '.yaml')
	if os.path.exists(dataset_default_path):
		with open(dataset_default_path) as file:
			dataset_defaults = yaml.load(file, Loader = yaml.Loader)
	meta_config = {}
	config = {}
	stage_config = {}

	config['dataset_name'] = raw_config['dataset']
	config['save_path'] = args.save_path

	config['image_channels'] = dataset_defaults['image_channels']
	config['image_size'] = raw_config.get('image_size', dataset_defaults.get('image_size'))
	for key in ['enc_gen_fc_layers', 'dis_cla_fc_layers', 'num_workers', 'device']:
		config[key] = raw_config.get(key, dataset_defaults.get(key, defaults[key]))
	config['device'] = args.device or config['device']
	config['data_path'] = args.data_path or raw_config.get('data_path')

	config['dataset_args'] = {}
	config['dataset_args'].update(dataset_defaults.get('dataset_args', {}))
	config['dataset_args'].update(raw_config.get('dataset_args', {}))

	config['all_factors'] = []
	for factor_defaults in dataset_defaults['factors']:
		if factor_defaults['name'] != 'unknown':
			config['all_factors'].append(factor_defaults['name'])

	config['plot_config'] = raw_config.get('plot_config', [])

	config['labeled_factors'] = []
	config['labeled_size'] = []
	config['labeled_keep_prob'] = []
	config['labeled_init'] = []
	if 'labeled_factors' in raw_config:
		for item in raw_config['labeled_factors']:
			if isinstance(item, str):
				name = item
				config['labeled_factors'].append(name)
				for factor_defaults in dataset_defaults['factors']:
					if factor_defaults['name'] == name:
						break
				size = factor_defaults['size']
				config['labeled_size'].append(size)
				config['labeled_keep_prob'].append(_make_keep_prob(size, factor_defaults.get('dropout')))
				config['labeled_init'].append(factor_defaults.get('init'))
			else:
				name = item['name']
				config['labeled_factors'].append(name)
				for factor_defaults in dataset_defaults['factors']:
					if factor_defaults['name'] == name:
						break
				size = item.get('size', factor_defaults['size'])
				config['labeled_size'].append(size)
				config['labeled_keep_prob'].append(_make_keep_prob(size, item.get('dropout', factor_defaults.get('dropout'))))
				config['labeled_init'].append(item.get('init', factor_defaults.get('init')))
	else:
		for factor_defaults in dataset_defaults['factors']:
			name = factor_defaults['name']
			if name != 'unknown':
				config['labeled_factors'].append(name)
				size = factor_defaults['size']
				config['labeled_size'].append(size)
				config['labeled_keep_prob'].append(_make_keep_prob(size, factor_defaults.get('dropout')))
				config['labeled_init'].append(factor_defaults.get('init'))

	if 'unknown_size' in raw_config:
		config['unknown_size'] = raw_config['unknown_size']
		config['unknown_keep_prob'] = _make_keep_prob(config['unknown_size'], raw_config.get('unknown_dropout', defaults['unknown_dropout']))
	else:
		config['unknown_size'] = 0
		unknown_keep_prob = []
		for factor_defaults in dataset_defaults['factors']:
			if factor_defaults['name'] not in config['labeled_factors']:
				size = factor_defaults['size']
				config['unknown_size'] += size
				unknown_keep_prob.extend(_make_keep_prob(size, factor_defaults.get('dropout')))
		if 'unknown_dropout' in raw_config:
			config['unknown_keep_prob'] = _make_keep_prob(config['unknown_size'], raw_config['unknown_dropout'])
		else:
			config['unknown_keep_prob'] = sorted(unknown_keep_prob, reverse = True)

	config['conv_channels'] = raw_config.get('conv_channels', dataset_defaults.get('conv_channels'))
	if config['conv_channels'] is None:
		num_levels = 0
		size = config['image_size']
		while size >= 12:
			size = (size + 2) // 4 * 2
			num_levels += 1
		if size >= 6:
			num_levels += 1
		config['conv_channels'] = defaults['conv_channels'][num_levels]
	else:
		num_levels = len(config['conv_channels'])

	config['conv_layers'] = raw_config.get('conv_layers', dataset_defaults.get('conv_layers', [1] * num_levels))
	config['fc_features'] = raw_config.get('fc_features', dataset_defaults.get('fc_features'))
	if config['fc_features'] is None:
		config['fc_features'] = defaults['fc_features'][min(num_levels, 8)]

	if config['image_size'] <= 32:
		batch_size = 64
	elif config['image_size'] <= 512:
		batch_size = 2048 // config['image_size']
	else:
		batch_size = 4
	
	for stage in ['stage1', 'classifier', 'stage2', 'lenc']:
		stage_config[stage] = {'batch_size': batch_size}
		stage_config[stage].update(defaults.get('all_stages', {}))
		stage_config[stage].update(defaults.get(stage, {}))
		stage_config[stage].update(dataset_defaults.get('all_stages', {}))
		stage_config[stage].update(dataset_defaults.get(stage, {}))
		stage_config[stage].update(raw_config.get('all_stages', {}))
		stage_config[stage].update(raw_config.get(stage, {}))

	if 'sample_grid_size' in raw_config:
		config['sample_row'] = raw_config['sample_grid_size']
	elif config['image_size'] <= 42:
		config['sample_row'] = 24
	elif config['image_size'] <= 128:
		config['sample_row'] = 512 // config['image_size'] * 2
	elif config['image_size'] <= 256:
		config['sample_row'] = 6
	else:
		config['sample_row'] = 4

	meta_config['has_unknown'] = config['unknown_size'] > 0
	meta_config['has_labeled'] = len(config['labeled_factors']) > 0
	meta_config['has_lenc_stage'] = meta_config['has_labeled'] and stage_config['stage2']['use_embedding']
	config['sample_col'] = config['sample_row'] // 2 if meta_config['has_unknown'] and meta_config['has_labeled'] else config['sample_row']

	if args.start_from is not None:
		meta_config['start_stage'] = args.start_from[0]
		meta_config['start_iter'] = int(args.start_from[1]) if len(args.start_from) > 1 else 0
	else:
		meta_config['start_stage'] = None

	meta_config['config'] = config
	meta_config['stage_config'] = stage_config

	return meta_config