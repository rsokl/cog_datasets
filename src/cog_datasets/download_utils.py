from pathlib import Path

import numpy as np


def _md5_check(fname):
    """Reads in data from disk and returns md5 hash"""
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_svhn(path, tmp_dir):
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    train_md5 = "e26dedcc434d2e4c54c9b2d4a06d8373"
    test_md5 = "eb5a983be6a315427106f1b164d9cef3"
    tmp_file_train = "__tmp_svhn_train.bin"
    tmp_file_test = "__tmp_svhn_test.bin"

    import os
    import urllib.request

    path = Path(path) / "svhn-python.npz"
    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    def extract(file, end_offset):
        import zlib

        with open(file, "rb") as f:
            f = f.read()
            endianness = "big" if f[126:128] == b"MI" else "little"
            num_bytes = int.from_bytes(f[132:136], endianness)
            uncompressed = zlib.decompress(f[136 : num_bytes + 136])
            data = (
                np.frombuffer(uncompressed[64:], dtype=np.uint8)
                .reshape(-1, 3, 32, 32)
                .transpose(0, 1, 3, 2)
            )

            uncompressed = zlib.decompress(f[num_bytes + 136 + 8 :])
            labels = np.frombuffer(uncompressed[56:end_offset], dtype=np.uint8)
        return data, labels

    print("Downloading from: {}".format(train_url))
    try:
        with urllib.request.urlopen(train_url) as response:
            with open(tmp_file_train, "wb") as handle:
                handle.write(response.read())

            assert (
                _md5_check(tmp_file_train) == train_md5
            ), "md5 checksum did not match!.. deleting file"

            train_data, train_labels = extract(tmp_file_train, -7)
            train_labels = train_labels.copy()
            train_labels[train_labels == 10] = 0
    finally:
        if os.path.isfile(tmp_file_train):
            os.remove(tmp_file_train)

    print("Downloading from: {}".format(test_url))
    try:
        with urllib.request.urlopen(test_url) as response:
            with open(tmp_file_test, "wb") as handle:
                handle.write(response.read())

            assert (
                _md5_check(tmp_file_test) == test_md5
            ), "md5 checksum did not match!.. deleting file"

            test_data, test_labels = extract(tmp_file_test, None)
            test_labels = test_labels.copy()
            test_labels[test_labels == 10] = 0
    finally:
        if os.path.isfile(tmp_file_test):
            os.remove(tmp_file_test)

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(
            f,
            x_train=train_data,
            y_train=train_labels,
            x_test=test_data,
            y_test=test_labels,
        )
    return


def _download_cifar100(path, tmp_dir):
    server_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    md5_checksum = "eb9058c3a382ffc7106e4002c42a8d85"
    tmp_file = "__tmp_cifar100.bin"

    import os
    import tarfile
    import urllib.request

    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    path = Path(path) / "cifar-100-python.npz"

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    print("Downloading from: {}".format(server_url))

    try:
        with urllib.request.urlopen(server_url) as response:
            with open(tmp_file, "wb") as handle:
                handle.write(response.read())

            assert (
                _md5_check(tmp_file) == md5_checksum
            ), "md5 checksum did not match!.. deleting file"

            with tarfile.open(tmp_file, "r:gz") as f:
                f.extractall(tmp_dir)
    finally:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)

    d = unpickle(tmp_dir / "cifar-100-python/train")
    train = d[b"data"]
    train_labels = d[b"fine_labels"]

    train = train.reshape(-1, 3, 32, 32)
    train_labels = np.asarray(train_labels)

    print("Writing train data:")
    print("Images: ", train.shape, train.dtype)
    print("Labels: ", train_labels.shape, train_labels.dtype)

    d = unpickle(tmp_dir / "cifar-100-python/test")
    test = np.asarray(d[b"data"]).reshape(-1, 3, 32, 32)
    test_labels = np.array(d[b"fine_labels"])

    print("Writing test data:")
    print("Images: ", test.shape, test.dtype)
    print("Labels: ", test_labels.shape, test_labels.dtype)

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(
            f, x_train=train, y_train=train_labels, x_test=test, y_test=test_labels
        )
    return


def _download_cifar10(path, tmp_dir):
    server_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    md5_checksum = "c58f30108f718f92721af3b95e74349a"
    tmp_file = "__tmp_cifar10.bin"

    import os
    import tarfile
    import urllib.request

    def unpickle(file):
        import pickle

        with open(file, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    path = Path(path) / "cifar-10-python.npz"

    if path.is_file():
        print("File already exists:\n\t{}".format(path))
        return None

    train = np.empty((50000, 3072), dtype=np.uint8)

    print("Downloading from: {}".format(server_url))

    try:
        with urllib.request.urlopen(server_url) as response:
            with open(tmp_file, "wb") as handle:
                handle.write(response.read())

            assert (
                _md5_check(tmp_file) == md5_checksum
            ), "md5 checksum did not match!.. deleting file"

            with tarfile.open(tmp_file, "r:gz") as f:
                f.extractall(tmp_dir)
    finally:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)

    train_labels = []
    for i in range(1, 6):
        d = unpickle(tmp_dir / "cifar-10-batches-py/data_batch_{}".format(i))
        train[(i - 1) * 10000 : i * 10000] = np.asarray(d[b"data"])
        train_labels += d[b"labels"]

    train = train.reshape(-1, 3, 32, 32)
    train_labels = np.asarray(train_labels)

    print("Writing train data:")
    print("Images: ", train.shape, train.dtype)
    print("Labels: ", train_labels.shape, train_labels.dtype)

    d = unpickle(tmp_dir / "cifar-10-batches-py/test_batch")
    test = np.asarray(d[b"data"]).reshape(-1, 3, 32, 32)
    test_labels = np.array(d[b"labels"])

    print("Writing test data:")
    print("Images: ", test.shape, test.dtype)
    print("Labels: ", test_labels.shape, test_labels.dtype)

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(
            f, x_train=train, y_train=train_labels, x_test=test, y_test=test_labels
        )
    return


def _download_mnist(path, server_url, tmp_file, check_sums):
    import gzip
    import os
    import urllib

    urls = dict(
        tr_img="train-images-idx3-ubyte.gz",
        tr_lbl="train-labels-idx1-ubyte.gz",
        te_img="t10k-images-idx3-ubyte.gz",
        te_lbl="t10k-labels-idx1-ubyte.gz",
    )

    data = {}
    for type_ in ["tr", "te"]:
        img_key = type_ + "_img"
        lbl_key = type_ + "_lbl"
        print("Downloading from: {}".format(server_url + urls[img_key]))
        with urllib.request.urlopen(server_url + urls[img_key]) as response:
            try:
                with open(tmp_file, "wb") as handle:
                    handle.write(response.read())

                # check md5
                expected = check_sums[urls[img_key]]
                found = _md5_check(tmp_file)
                msg = "md5 checksum did not match!.. deleting file:\nexpected: {}\nfound: {}".format(
                    expected, found
                )
                assert expected == found, msg

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp = np.frombuffer(uncompressed.read(), dtype=np.uint8, offset=16)
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        print("Downloading from: {}".format(server_url + urls[lbl_key]))
        with urllib.request.urlopen(server_url + urls[lbl_key]) as response:
            try:
                with open(tmp_file, "wb") as handle:
                    handle.write(response.read())

                # check md5
                expected = check_sums[urls[lbl_key]]
                found = _md5_check(tmp_file)
                msg = "md5 checksum did not match!.. deleting file:\nexpected: {}\nfound: {}".format(
                    expected, found
                )
                assert expected == found, msg

                with gzip.open(tmp_file, "rb") as uncompressed:
                    tmp_lbls = np.frombuffer(
                        uncompressed.read(), dtype=np.uint8, offset=8
                    )
            finally:
                if os.path.isfile(tmp_file):
                    os.remove(tmp_file)

        data[img_key] = tmp.reshape(tmp_lbls.shape[0], 1, 28, 28)
        data[lbl_key] = tmp_lbls

    print("Saving to: {}".format(path))
    with path.open(mode="wb") as f:
        np.savez_compressed(
            f,
            x_train=data["tr_img"],
            y_train=data["tr_lbl"],
            x_test=data["te_img"],
            y_test=data["te_lbl"],
        )
