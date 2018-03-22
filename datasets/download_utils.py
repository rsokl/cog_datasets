import numpy as np
from pathlib import Path
from os import path

__all__ = ["download_cifar10", "download_fashion_mnist"]

_path = Path(path.dirname(path.abspath(__file__)))


def _md5_check(fname):
    """ Reads in data from disk and returns md5 hash"""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_cifar10(path=None):
    """ Download the cifar-10 dataset and save it as a .npz archive.
        md5 check-sum verficiation is performed.

        Parameters
        ----------
        path : Optional[str, pathlib.Path]
            path to .npz file to be saved (including the filename itself)

            if `None`, path = path/to/DataSets/datasets/cifar-10-python.npz"""

    def _download_cifar10(path=None, tmp_dir=None):
        server_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        md5_checksum = "c58f30108f718f92721af3b95e74349a"
        tmp_dir = Path(".") / "_tmp_dir"
        tmp_file = "__tmp_cifar10.bin"

        import tarfile
        import urllib.request
        import os

        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')

        path = Path(path) if path is not None else _path / 'cifar-10-python.npz'

        if path.is_file():
            print("File already exists:\n\t{}".format(path))
            return None

        train = np.empty((50000, 3072), dtype=np.uint8)
        test = np.empty((10000, 3072), dtype=np.uint8)

        print("Downloading from: {}".format(server_url))

        try:
            with urllib.request.urlopen(server_url) as response:
                with open(tmp_file, 'wb') as handle:
                    handle.write(response.read())

                assert _md5_check(tmp_file) ==  md5_checksum, "md5 checksum did not match!.. deleting file"

                with tarfile.open(tmp_file, 'r:gz') as f:
                    f.extractall(tmp_dir)
        finally:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)

        train_labels = []
        for i in range(1, 6):
            d = unpickle(tmp_dir / "cifar-10-batches-py/data_batch_{}".format(i))
            train[(i - 1)*10000:i*10000] = np.asarray(d[b'data'])
            train_labels += d[b'labels']

        train = train.reshape(-1, 3, 32, 32)
        train_labels = np.asarray(train_labels)

        print("Writing train data:")
        print("Images: ", train.shape, train.dtype)
        print("Labels: ", train_labels.shape, train_labels.dtype)

        d = unpickle(tmp_dir / "cifar-10-batches-py/test_batch")
        test = np.asarray(d[b'data']).reshape(-1, 3, 32, 32)
        test_labels = np.array(d[b'labels'])

        print("Writing test data:")
        print("Images: ", test.shape, test.dtype)
        print("Labels: ", test_labels.shape, test_labels.dtype)

        print("Saving to: {}".format(path))
        with path.open(mode="wb") as f:
            np.savez_compressed(f, x_train=train, y_train=train_labels,
                                x_test=test, y_test=test_labels)
        return

    import os
    import shutil
    tmp_dir = Path(".") / "_tmp_dir_"
    if tmp_dir.exists():
        print("Directory: {} already exists - an intermediate directory needs to be constructed here".format(tmp_dir))
        print("move/delete that directory and try again.")
        return None

    try:
        _download_cifar10(path, tmp_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def download_fashion_mnist(path=None):
    """ Function for downloading fashion-mnist and saves fashion-mnist as a
        numpy compressed-archive. md5 check-sum verficiation is performed.

        Parameters
        ----------
        path : Optional[pathlib.Path, str]
            Path to containing .npz file. If `None`, the path to the DataSets module is used."""
    import urllib.request
    import gzip
    import os

    path = Path(path) if path is not None else _path / "fashion_mnist.npz"

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    if path.is_dir():
        print("`path` specifies a directory. It should specify the file-destination, including the file-name.")
        return None

    server_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    urls = dict(tr_img="train-images-idx3-ubyte.gz", tr_lbl="train-labels-idx1-ubyte.gz",
                te_img="t10k-images-idx3-ubyte.gz", te_lbl="t10k-labels-idx1-ubyte.gz")

    check_sums = {"train-images-idx3-ubyte.gz" : "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
                  "train-labels-idx1-ubyte.gz" : "25c81989df183df01b3e8a0aad5dffbe",
                  "t10k-images-idx3-ubyte.gz" : "bef4ecab320f06d8554ea6380940ec79",
                  "t10k-labels-idx1-ubyte.gz" : "bb300cfdad3c16e7a12a480ee83cd310"}
    data = {}
    tmp_file = '__tmp_fashion_mnist.bin'  # write file to perform md5 check-sum
    for type_ in ["tr", "te"]:
        img_key = type_ + "_img"
        lbl_key = type_ + "_lbl"
        print("Downloading from: {}".format(server_url + urls[img_key]))
        with urllib.request.urlopen(server_url + urls[img_key]) as response:
            try:
                with open(tmp_file, 'wb') as handle:
                    handle.write(response.read())

                assert _md5_check(tmp_file) == check_sums[urls[img_key]] , "md5 checksum did not match!.. deleting file"

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=16)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        print("Downloading from: {}".format(server_url + urls[lbl_key]))
        with urllib.request.urlopen(server_url + urls[lbl_key]) as response:
            try:
                with open(tmp_file, 'wb') as handle:
                    handle.write(response.read())

                assert _md5_check(tmp_file) == check_sums[urls[lbl_key]] , "md5 checksum did not match!.. deleting file"

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp_lbls = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=8)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        data[img_key] = tmp.reshape(tmp_lbls.shape[0], 1, 28, 28)

        # labels -> one-hot
        data[lbl_key] = np.zeros((tmp_lbls.shape[0], 10), dtype="uint8")
        data[lbl_key][range(tmp_lbls.shape[0]), tmp_lbls] = 1

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(f, x_train=data["tr_img"], y_train=data["tr_lbl"],
                            x_test=data["te_img"], y_test=data["te_lbl"])
