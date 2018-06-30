import numpy as np
from pathlib import Path


def _md5_check(fname):
    """ Reads in data from disk and returns md5 hash"""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_cifar10(path, tmp_dir):
    server_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    md5_checksum = "c58f30108f718f92721af3b95e74349a"
    tmp_file = "__tmp_cifar10.bin"

    import tarfile
    import urllib.request
    import os

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')

    path = Path(path) / 'cifar-10-python.npz'

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    train = np.empty((50000, 3072), dtype=np.uint8)

    print("Downloading from: {}".format(server_url))

    try:
        with urllib.request.urlopen(server_url) as response:
            with open(tmp_file, 'wb') as handle:
                handle.write(response.read())

            assert _md5_check(tmp_file) == md5_checksum, "md5 checksum did not match!.. deleting file"

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


def _download_mnist(path, server_url, tmp_file, check_sums=None):
    import urllib
    import gzip
    import os
    urls = dict(tr_img="train-images-idx3-ubyte.gz", tr_lbl="train-labels-idx1-ubyte.gz",
                te_img="t10k-images-idx3-ubyte.gz", te_lbl="t10k-labels-idx1-ubyte.gz")

    data = {}
    for type_ in ["tr", "te"]:
        img_key = type_ + "_img"
        lbl_key = type_ + "_lbl"
        print("Downloading from: {}".format(server_url + urls[img_key]))
        with urllib.request.urlopen(server_url + urls[img_key]) as response:
            try:
                with open(tmp_file, 'wb') as handle:
                    handle.write(response.read())

                if check_sums is not None and isinstance(check_sums[urls[img_key]], str):
                    assert _md5_check(tmp_file) == check_sums[urls[img_key]], "md5 checksum did not match!.. deleting file"
                elif check_sums is not None and isinstance(check_sums[urls[img_key]], int):
                    os.path.getsize(tmp_file) == check_sums[urls[img_key]], "downloaded filesize is bad!.. deleting file"

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

                if check_sums is not None and isinstance(check_sums[urls[img_key]], str):
                    # check md5
                    assert _md5_check(tmp_file) == check_sums[urls[img_key]], "md5 checksum did not match!.. deleting file"
                elif check_sums is not None and isinstance(check_sums[urls[img_key]], int):
                    # check filesize
                    os.path.getsize(tmp_file) == check_sums[urls[img_key]], "downloaded filesize is bad!.. deleting file"

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp_lbls = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=8)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        data[img_key] = tmp.reshape(tmp_lbls.shape[0], 1, 28, 28)
        data[lbl_key] = tmp_lbls

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(f, x_train=data["tr_img"], y_train=data["tr_lbl"],
                            x_test=data["te_img"], y_test=data["te_lbl"])
