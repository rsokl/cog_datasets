""" 
Save machine learning data sets to a common location, and load them without
having to specify a path.

All datasets will be saved to `datasets.path`. By default, this will be point to
~/datasets. You can update this path via `datasets.set_path`.
"""

from pathlib import Path

import numpy as np

from . import _version
from .toydata import ToyData

__version__ = _version.get_versions()["version"]

__all__ = [
    "load_cifar10",
    "load_cifar100",
    "load_mnist",
    "load_svhn",
    "load_fashion_mnist",
    "download_cifar10",
    "download_cifar100",
    "download_fashion_mnist",
    "download_mnist",
    "download_svhn",
    "ToyData",
]


_config_file = Path.home() / ".datasets"


def get_path(verbose=False):
    if _config_file.is_file():
        with _config_file.open("r") as f:
            header, path = f.readlines()
        path = Path(path)
    else:
        path = Path.home() / "datasets"
        path.mkdir(exist_ok=True)
    if verbose:
        print("`datasets module: datasets will be loaded from '{}'".format(path))
    return path


get_path(verbose=True)


def set_path(new_path, mkdir=False):
    """
    Specify the path to which datasets will be saved. This path is saved to
    a config file saved at: ~/.datasets

    Parameters
    ----------
    new_path : PathLike
        The path to the directory to which all datasets will be saved.

    mkdir : bool, optional (default=False)
        If `True` the specified directory will be created if it doesn't already exist.
    """
    new_path = Path(new_path)
    if mkdir:
        new_path.mkdir(exist_ok=True)
    with _config_file.open(mode="w") as f:
        f.write(
            "# The python pacakge `datasets` will write data to the following directory:\n"
        )
        f.write(str(new_path.absolute()))
    get_path(verbose=True)


def restore_default_path(are_you_sure):
    """
    Deletes ~/.datasets config file and restores the path to '~/datasets

    Parameters
    ----------
    are_you_sure : bool
        Users must explicitly specify `True` to reset the path.
    """
    global path
    import os

    if are_you_sure is not True:
        print(
            "You must explicitly specify `restore_default_path(True)` to reset the path."
        )
        return

    if _config_file.is_file():
        os.remove(_config_file)
    path = get_path(verbose=False)


def download_svhn():
    """Download the streetview house numbers dataset and save it as a .npz archive.
    md5 check-sum verification is performed.

    path = <path_to_datasets>/svhn-python.npz"""
    import shutil

    from cog_datasets.download_utils import _download_svhn

    tmp_dir = get_path() / "_tmp_dir_"
    if tmp_dir.exists():
        print(
            "Directory: {} already exists - an intermediate directory needs to be constructed here".format(
                tmp_dir
            )
        )
        print("move/delete that directory and try again.")
        return None

    try:
        _download_svhn(get_path(), tmp_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def download_cifar10():
    """Download the cifar-10 dataset and save it as a .npz archive.
    md5 check-sum verification is performed.

    path = <path_to_datasets>/cifar-10-python.npz"""
    import shutil

    from cog_datasets.download_utils import _download_cifar10

    tmp_dir = get_path() / "_tmp_dir_"
    if tmp_dir.exists():
        print(
            "Directory: {} already exists - an intermediate directory needs to be constructed here".format(
                tmp_dir
            )
        )
        print("move/delete that directory and try again.")
        return None

    try:
        _download_cifar10(get_path(), tmp_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def download_cifar100():
    """Download the cifar-100 dataset and save it as a .npz archive.
    md5 check-sum verification is performed.

    path = <path_to_datasets>/cifar-100-python.npz"""
    import shutil

    from cog_datasets.download_utils import _download_cifar100

    tmp_dir = get_path() / "_tmp_dir_"
    if tmp_dir.exists():
        print(
            "Directory: {} already exists - an intermediate directory needs to be constructed here".format(
                tmp_dir
            )
        )
        print("move/delete that directory and try again.")
        return None

    try:
        _download_cifar100(get_path(), tmp_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def download_fashion_mnist():
    """Function for downloading fashion-mnist and saves fashion-mnist as a
    numpy compressed-archive. md5 check-sum verficiation is performed.

    Parameters
    ----------
    path : Optional[pathlib.Path, str]
        Path to containing .npz file. If `None`, the path to the DataSets module is used."""
    from cog_datasets.download_utils import _download_mnist, _md5_check

    path = get_path() / "fashion_mnist.npz"
    tmp_file = get_path() / "__mnist.bin"

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    if path.is_dir():
        print(
            "`path` specifies a directory. It should specify the file-destination, including the file-name."
        )
        return None

    server_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    check_sums = {
        "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
    }
    _download_mnist(
        path, server_url=server_url, tmp_file=tmp_file, check_sums=check_sums
    )


def download_mnist():
    """Function for downloading mnist and saves fashion-mnist as a
    numpy compressed-archive. file-size verificiation is performed."""

    from cog_datasets.download_utils import _download_mnist, _md5_check

    path = get_path() / "mnist.npz"
    tmp_file = get_path() / "__mnist.bin"

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    if path.is_dir():
        print(
            "`path` specifies a directory. It should specify the file-destination, including the file-name."
        )
        return None

    server_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    md5_hashes = {
            "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
            "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
            "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }

    _download_mnist(
        path, server_url=server_url, tmp_file=tmp_file, check_sums=md5_hashes
    )


def load_svhn(fname="svhn-python.npz"):
    """The SVHN dataset consists of 99289x3x32x32 uint-8 color images in 10
    classes. There are 73257 training images and 26032 test images.

    The labels are integers in [0, 9]

    http://ufldl.stanford.edu/housenumbers/

    Parameters
    ----------
    fname : str, optional (default="cifar-10-python.npz")
        The filename of the .npz archive storing the cifar-10 data

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        training-data, training-labels, test-data, test-labels

    Notes
    -----
    A tuple of the categories corresponding to the data's integer labels are bound as an
    attribute of this function:

        `dataset.load_svhn.labels`
    """

    path = get_path()
    if not (path / fname).exists():
        msg = """ Data not found! Please download the data (svhn-python.npz) using 
                 `datasets.download_svhn()`"""
        raise FileNotFoundError(msg)

    with np.load(str(path / fname)) as data:
        xtr, ytr, xte, yte = tuple(
            data[key] for key in ["x_train", "y_train", "x_test", "y_test"]
        )
    print("svhn loaded")
    return xtr, ytr, xte, yte


load_svhn.labels = tuple(str(i) for i in range(10))


def load_cifar10(fname="cifar-10-python.npz"):
    """The CIFAR-10 dataset consists of 60000x3x32x32 uint-8 color images in 10
    classes, with 6000 images per class. There are 50000 training images
    and 10000 test images.

    The labels are integers in [0, 9]

    https://www.cs.toronto.edu/~kriz/cifar.html

    Parameters
    ----------
    fname : str, optional (default="cifar-10-python.npz")
        The filename of the .npz archive storing the cifar-10 data

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

    path = get_path()
    if not (path / fname).exists():
        msg = """ Data not found! Please download the data (cifar-10-python.npz) using 
                 `datasets.download_cifar10()`"""
        raise FileNotFoundError(msg)

    with np.load(str(path / fname)) as data:
        xtr, ytr, xte, yte = tuple(
            data[key] for key in ["x_train", "y_train", "x_test", "y_test"]
        )
    print("cifar-10 loaded")
    return xtr, ytr, xte, yte


load_cifar10.labels = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar100(fname="cifar-100-python.npz"):
    """The CIFAR-100 dataset consists of 60000x3x32x32 uint-8 color images in 100
    classes, with 600 images per class. There are 50000 training images
    and 10000 test images.

    The labels are integers in [0, 99]

    https://www.cs.toronto.edu/~kriz/cifar.html

    Parameters
    ----------
    fname : str, optional (default="cifar-100-python.npz")
        The filename of the .npz archive storing the cifar-100 data.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        training-data, training-labels, test-data, test-labels

    Notes
    -----
    A tuple of the categories corresponding to the data's integer labels are bound as an
    attribute of this function:

        `dataset.load_cifar100.labels`
    """

    path = get_path()
    if not (path / fname).exists():
        msg = """ Data not found! Please download the data (cifar-100-python.npz) using 
                 `datasets.download_cifar100()`"""
        raise FileNotFoundError(msg)

    with np.load(str(path / fname)) as data:
        xtr, ytr, xte, yte = tuple(
            data[key] for key in ["x_train", "y_train", "x_test", "y_test"]
        )
    print("cifar-100 loaded")
    return xtr, ytr, xte, yte


load_cifar100.labels = (
    "apples",
    "aquarium fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottles",
    "bowls",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "cans",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "computer keyboard",
    "couch",
    "crab",
    "crocodile",
    "cups",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "lamp",
    "lawn-mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple",
    "motorcycle",
    "mountain",
    "mouse",
    "mushrooms",
    "oak",
    "oranges",
    "orchids",
    "otter",
    "palm",
    "pears",
    "pickup truck",
    "pine",
    "plain",
    "plates",
    "poppies",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "roses",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflowers",
    "sweet peppers",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulips",
    "turtle",
    "wardrobe",
    "whale",
    "willow",
    "wolf",
    "woman",
    "worm",
)


def load_fashion_mnist(fname="fashion_mnist.npz"):
    """Loads the fashion-mnist dataset (including train & test, along with their labels).

    The data set is loaded as Nx1x28x28 uint8 numpy arrays. N is the size of the
    data set - N=60,000 for the training set, and N=10,000 for the test set.

    The labels are integers in [0, 9]

    Additional information regarding the fashion-mnist data set can be found here:
        - https://github.com/zalandoresearch/fashion-mnist

    Parameters
    ----------
    fname : str, optional (default="fashion_mnist.npz")
        The filename of the .npz file to be loaded

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

    path = get_path()

    if not (path / fname).exists():
        import inspect

        msg = """ Data not found! Please download the data (fashion_mnist.npz) using 
                 `datasets.download_fashion_mnist()`"""
        raise FileNotFoundError(inspect.cleandoc(msg))

    with np.load(str(path / fname)) as data:
        out = [data[key] for key in ["x_train", "y_train", "x_test", "y_test"]]

    print("fashion-mnist loaded")
    return tuple(out)


load_fashion_mnist.labels = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_mnist(fname="mnist.npz"):
    """The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of
    10,000 examples. It is a subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image.

    The data set is loaded as Nx1x28x28 uint8 numpy arrays. N is the size of the
    data set - N=60,000 for the training set, and N=10,000 for the test set.

    The labels are integers in [0, 9]

    http://yann.lecun.com/exdb/mnist/

    Parameters
    ----------
    fname : str, optional (default="mnist.npz")
        The filename of the .npz archive storing the mnist data

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        training-data, training-labels, test-data, test-labels"""
    path = get_path()
    with np.load(path / fname) as data:
        out = tuple(
            data[str(key)] for key in ["x_train", "y_train", "x_test", "y_test"]
        )
    print("mnist loaded")
    return out


load_mnist.labels = tuple(str(i) for i in range(10))
