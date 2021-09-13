# DisUnknown: Distilling Unknown Factors for Disentanglement Learning

## Requirements

- PyTorch >= 1.8.0
    - [`torch.linalg.eigh()`](https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html#torch.linalg.eigh) is used in a few places. If you use a older version please change them to [`torch.symeig()`](https://pytorch.org/docs/stable/generated/torch.symeig.html#torch.symeig)
- [PyYAML](https://pyyaml.org/), for loading configuration files
- Optional: [h5py](https://www.h5py.org/), for using the 3D Shapes dataset
- Optional: [Matplotlib](https://matplotlib.org/stable/index.html), for plotting sample distributions in code space

## Preparing Datasets

Dataset classes and default configurations are provided for the following datasets. See below for how to add new datasets, or you can open an issue and the author might consider adding it. Some datasets need to be prepared before using:

```
$ python disentangler.py prepare_data <dataset_name> --data_path </path/to/dataset>
```

If the dataset does not have a standard training/test split it will be split randomly. Use the `--test_portion <portion>` option to set the portion of test samples. Some dataset have additional options.

- MNIST, Fashion-MNIST, QMNIST, SVHN
    - Dataset names are `mnist`, `fashion_mnist`, `qmnist`, `svhn`.
    - `data_path` should be the same as those for the built-in dataset classes provided by torchvision.
    - We use the full NIST digit dataset from QMNIST (`what == 'nist'`) and it needs to be split.
- [3D Chairs](https://www.di.ens.fr/willow/research/seeing3Dchairs/)
    - Dataset name is `chairs`.
    - `data_path` should be the folder containing the `rendered_chairs` folder.
    - Needs to be split.
    - You may use `--compress` to down-sample and compress the images and save them as a NumPy array of PNG-encoded `ByteArray`s. Use `--downsample_size <size>` to set image size, defaults to 128. Note that this does not dictate the training-time image size, which is configured separately. Compressing the images speeds up training only slightly if multi-process dataloader is used but makes plotting significantly faster.
    - Unrelated to this work, but the author wants to tell you that this dataset curiously contains 31 azimuth angles and two altitudes for a total of 62 images for each chair with image id `031` skipped, apparently because 32 was the intended number of azimuth angles but when they rendered the images those were generated using `np.linspace(0, 360, 32)`, ignoring the fact that 0 and 360 are the same angle, then removed the duplicating images `031` and `063` after they realized the mistake. Beware of off-by-one errors in linspace, especially if it is also circular!
- [3D shapes](https://github.com/deepmind/3d-shapes)
    - Dataset name is `3dshapes`.
    - `data_path` should be the folder containing `3dshapes.h5`.
    - Needs to be split.
    - You may use `--extract` to extract all images and then save them as a NumPy array of PNG-encoded `ByteArray`s. This is mainly for space-saving: the original dataset, when extracted from HDFS, takes 5.9GB of memory. The re-compressed version takes 2.2GB. Extraction and compression takes about an hour.
- The author would also like to note that he did not use the [dSprites](https://github.com/deepmind/dsprites-dataset/) dataset mainly because two of the three shapes have rotation symmetry which causes some different angles to produce the same image, i.e. the correct factor value for an image might not be unique, which is rather problematic. Furthermore, each shape has a different symmetry, so that this cannot be solved by just merging some angle values, which complicates things further.

## Training

To train, use

```
$ python disentangler.py train --config_file </path/to/config/file> --save_path </path/to/save/folder>
```

The configuration file is in YAML. See the commented example for explanations. If `config_file` is omitted, it is expected that `save_path` already exists and contains `config.yaml`. Otherwise `save_path` will be created if it does not exist, and `config_file` will be copied into it. If `save_path` already contains a previous training run that has been halted, it will by default resume from the latest checkpoint. `--start_from <stage_name> [<iteration>]` can be used to choose another restarting point. `--start_from stage1` to restart from scratch. Specifying `--data_path` or `--device` will override those settings in the configuration file.

Although our goal is to deal with the cases where some factors are labeled and some factors are unknown, it feels wrong not to extrapolate to the cases where all factors are labeled or where all factors are unknown. Wo do allow these, but some parts of our method will be unnecessary and will be discarded accordingly. In particular if all factors are unknown then we just train a VAE in stage I and then a GAN having the same code space in stage II, so you can use this code for just training a GAN. We don't have the myriad of GAN tricks though.

## Meaning of Visualization Images

During training, images generated for visualization will be saved in subfolder `samples`. `test_images.jpg` contains images from the test set in even-numbered columns (starting from zero), with odd-numbered columns empty. The generated images will contain corresponding reconstructed samples in even-numbered columns, while each image in odd-numbered columns is generated by combining the unknown code from its left and the labeled code from its right (warp to the next row). 

## Adding a New Dataset

`__init__()` should accept four positional arguments `root`, `part`, `labeled_factors`, `transform` in that order, plus any additional keyword arguments that one expects to receive from `dataset_args` in the configuration file. `root` is the path to the dataset folder. `transform` is as usual. `part` can be `train`, `test` or `plot`, specifying which subset of the dataset to load. The plotting set is generally the same as the test set, but a separate `part` is passed in so that it can be smaller if the test set is too large.

`labeled_factors` is a list of factor names. `__getitem__()` should return a tuple `(image, labels)` where `image` is a PIL image and `labels` is a one-dimensional PyTorch tensor of type `torch.int64`, containing the labels for that image in the order listed in `labeled_factors`. `labels` should always be a one-dimensional tensor even if there is only one labeled factor, not a Python `int` or a zero-dimensional tensor. If `labeled_factors` is empty then `__getitem__()` should only return `image` only.

In addition, metadata about the factors should be available in the following properties: `nclass` should be a list of `int`s containing the number of classes of each factor, and `class_freq` should be a list of PyTorch tensors, each being one-dimensional, containing the distribution of classes of each factor in (the current split of) the dataset.

If any preparation is required, implement a static method `prepare_data(args)` where `args` is the return value of `argparse.ArgumentParser.parse_args()`, containing properties `data_path` and `test_portion` by default. If additional command-line arguments are needed, implement a static method `add_prepare_args(parser)` where `parser.add_argument()` can be called.

Finally add it to the dictionary of recognized datasets in `data/__init__.py`.

Default configuration should also be created as `default_config/datasets/<dataset_name>.yaml`. It should at a minimum contain `image_size`, `image_channels` and `factors`. `factors` has the same syntax as `labeled_factors` as explained in the example training configuration. It should contain a complete list of all factors. In particular, if the dataset does not include a complete set of labels, there should be a factor called `unknown` which will become the default unknown factor if `labeled_factors` is not set in the training configuration.

Any additional settings in the default configuration will override global defaults in `default_config/default_config.yaml`.