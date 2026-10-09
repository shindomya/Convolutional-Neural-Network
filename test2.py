import numpy as np
from PIL import Image
import time
import requests
import urllib
import matplotlib
import gzip
import pickle
import sys
import os
#eeee
#start
time_start = time.perf_counter()
data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}

data_dir = "./_data"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True, timeout=10)
        resp.raise_for_status()
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

#images
for key in ("training_images","test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1,28 * 28)
#labels
for key in ("training_labels","test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key], "rb") as mnist_file:
                   mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)



#stop
end_time = time.perf_counter()

print(f"Code execution took: {end_time - time_start:.4f} seconds")
