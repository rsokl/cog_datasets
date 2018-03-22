""" Save machine learning data sets to a common location, and load them without
    having to specify a path.
"""

from os import path
from pathlib import Path
import numpy as np

from .download_utils import download_cifar10, download_fashion_mnist
from .toydata import ToyData

_path = Path(path.dirname(path.abspath(__file__)))


__all__ = ["load_cifar10", 
           "load_mnist", 
           "load_fashion_mnist", 
           "download_cifar10",
           "download_fashion_mnist", 
           "ToyData",
           "get_cifar10_path",
           "get_mnist_path",
           "get_fashion_mnist_path",
           ]


def _get_dataset_path(dataset_name):
    return _path / dataset_name

def get_cifar10_path(): return _get_dataset_path("cifar-10-python.npz")
def get_mnist_path(): return _get_dataset_path("mnist.npz")
def get_fashion_mnist_path(): return _get_dataset_path("fashion_mnist.npz")



def load_cifar10(fname='cifar-10-python.npz'):
    """ The CIFAR-10 dataset consists of 60000x3x32x32 color images in 10 
        classes, with 6000 images per class. There are 50000 training images 
        and 10000 test images.

        The labels are formatted using one-hot encoding.

        https://www.cs.toronto.edu/~kriz/cifar.html

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            training-data, training-labels, test-data, test-labels

        Notes
        -----
        A tuple of the categories corresponding to the data's integer labels are bound as an
        attribute of this function:

            `dataset.load_cifar10.labels`
        """
    

    if not (_path / fname).exists():
        msg = """ Data not found! Please download the data (cifar-10-python.npz) using 
                 `datasets.download_cifar10()`"""
        raise FileNotFoundError(msg)

    with np.load(str(_path / fname)) as data:
        xtr, ytr, xte, yte = tuple(data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test'])
    print("cifar-10 loaded")
    return xtr, ytr, xte, yte

load_cifar10.labels = ("airplane",
                       "automobile",
                       "bird",
                       "cat",
                       "deer",
                       "dog",
                       "frog",
                       "horse",
                       "ship",
                       "truck")


def load_mnist(fname="mnist.npz"):
    """ The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of
        10,000 examples. It is a subset of a larger set available from NIST. The digits have been
        size-normalized and centered in a fixed-size image.

        The labels are formatted using one-hot encoding.

        http://yann.lecun.com/exdb/mnist/

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            training-data, training-labels, test-data, test-labels"""
    with np.load(str(_path / fname)) as data:
        out = tuple(data[str(key)] for key in ['x_train', 'y_train', 'x_test', 'y_test'])
    print("mnist loaded")
    return out


def load_fashion_mnist(*, path=None, fname="fashion_mnist.npz", zero_pad=True):
    """ Loads the fashion-mnist dataset (including train & test, along with their labels).

        The data set is loaded as Nx1x28x28 or Nx1x32x32 numpy arrays. N is the size of the
        data set - N = 60,000 for the training set, and N = 10,000 for the test set.

        The labels are formatted using one-hot encoding.

        Given that mnist is often zero padded (symmetrically), so that its 28x28 images
        become 32x32 after zero-padding, this padding option is used by default.

        Additional information regarding the fashion-mnist data set can be found here:
            - https://github.com/zalandoresearch/fashion-mnist

        Parameters
        ----------
        path : Optional[str, pathlib.Path]
            Path to directory containing the .npz file. If `None`, the path to the DataSets module is used.

        fname : str, optional (default="fashion_mnist.npz")
            The filename of the .npz file to be loaded

        zero_pad : bool, optional (default=True)
            If True, apply symmetric zero-padding of depth 2, to each side of an image

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            training-data, training-labels, test-data, test-labels

        Notes
        -----
        A tuple of the categories corresponding to the data's integer labels are bound as an
        attribute of this function:

            `dataset.load_fashion_mnist.labels`
        """
    if path is None:
        path = _path

    elif isinstance(path, str):
        path = Path(path)

    if not (path / fname).exists():
        import inspect
        msg = """ Data not found! Please download the data (fashion_mnist.npz) using 
                 `datasets.download_fashion_mnist()`"""
        raise FileNotFoundError(inspect.cleandoc(msg))

    with np.load(str(path / fname)) as data:
        out = [data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test']]

    assert isinstance(zero_pad, bool)
    if zero_pad:
        for index in (0, 2):
            out[index] = np.pad(out[index], pad_width=((0, 0), (0, 0), (2, 2), (2, 2)),
                                mode="constant", constant_values=0)
    print("fashion-mnist loaded")
    return tuple(out)


load_fashion_mnist.labels = ('T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle boot')

