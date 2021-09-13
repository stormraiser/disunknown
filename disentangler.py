import argparse

import trainers
import data

from process_config import *

def make_parser():
	parser = argparse.ArgumentParser()

	# master options
	subparsers = parser.add_subparsers()

	train_parser = subparsers.add_parser('train')
	train_parser.set_defaults(func = train)

	train_parser.add_argument('--config_file',
		help = 'path to configuration file')
	train_parser.add_argument('--save_path',
		help = 'path to save folder')

	train_parser.add_argument('--data_path',
		help = 'path to dataset')
	train_parser.add_argument('--device',
		help = 'device')
	train_parser.add_argument('--start_from', nargs = '+',
		help = 'start stage and iteration')

	prep_parser = subparsers.add_parser('prepare_data')
	prep_parser.set_defaults(func = prepare)

	prep_parser.add_argument('dataset',
		help = 'dataset')
	prep_parser.add_argument('--data_path',
		help = 'path to dataset')
	prep_parser.add_argument('--test_portion', type = float, default = 0.1,
		help = 'portion of test samples')

	return parser

def train(args, extra_args):
	trainers.Trainer(process_config(args)).run()	

def prepare(args, extra_args):
	dataset_class = data.datasets[args.dataset]
	if hasattr(dataset_class, 'prepare_data'):
		dataset_parser = argparse.ArgumentParser()
		if hasattr(dataset_class, 'add_prepare_args'):
			dataset_class.add_prepare_args(dataset_parser)
		dataset_args = dataset_parser.parse_args(extra_args)
		args.__dict__.update(dataset_args.__dict__)
		dataset_class.prepare_data(args)
	else:
		print('Nothing to prepare for this dataset.')

args, extra_args = make_parser().parse_known_args()
args.func(args, extra_args)
