from .mnist import MNIST, FashionMNIST, EMNIST
from .qmnist import QMNIST
from .PairedDataset import PairedDataset
from .threedshapes import ThreeDShapes
from .dsprites import DSprites
from .chairs import Chairs
from .svhn import SVHN
from .cifar import CIFAR10

datasets = {
	'mnist': MNIST,
	'fashion_mnist': FashionMNIST,
	'emnist': EMNIST,
	'qmnist': QMNIST,
	'3dshapes': ThreeDShapes,
	'chairs': Chairs,
	'svhn': SVHN,
	'dsprites': DSprites,
	'cifar10': CIFAR10
}
