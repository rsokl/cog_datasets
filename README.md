# DataSets

Convenience package so that you can keep your data sets in one place, but load them anywhere.

## Installation Instructions
Clone Datasets, navigate to the resulting directory, and run

```shell
python setup.py develop
```

## Usage
```python
# utilities for downloading cifar-10 and fashion-mnist
# mnist-data is included in this repo
from datasets import download_cifar10, download_fashion_mnist

# utilities for loading various data sets
from datasets import load_mnist, load_fashion_mnist, load_cifar10, ToyData
```