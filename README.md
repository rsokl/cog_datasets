# DataSets

Convenience package so that you can keep your data sets in one place, but load them anywhere.

## Installation Instructions
Clone Datasets, navigate to the resulting directory, and run

```shell
python setup.py install
```

## Usage
```python
# utilities for downloading cifar-10, mnist, and fashion-mnist
# mnist-data is included in this repo
from datasets import download_cifar10, download_fashion_mnist, download_mnist

# utilities for loading various data sets
from datasets import load_mnist, load_fashion_mnist, load_cifar10, ToyData
```

By default, all data sets will be downloaded to ~/datasets. You can overwrite this via

```python
>>> import datasets
>>> datasets.set_path('mydir', mkdir=True)
`datasets module: datasets will be loaded from 'C:\Users\You\mydir'
```

This will write your `datasets` path to `~/.datasets`.

You can restore the default path with:

```python
>>> datasets.restore_default_path(True)
```
