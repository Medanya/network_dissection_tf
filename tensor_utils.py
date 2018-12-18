import numpy as np
from glob import glob
from tqdm import tqdm
import os


def calculate_percentiles(tensor_dir, save_path, percentile=99.5):
    percentiles = list()
    all_tensors = glob(os.path.join(tensor_dir, '*.npy'))

    assert len(all_tensors) > 0, "No npy files in tensor directory: " + tensor_dir
    for tensor_name in tqdm(all_tensors):
        tensor = np.load(tensor_name)
        tensor = tensor.reshape((-1, tensor.shape[2]))
        percentiles.append(np.percentile(tensor, percentile, axis=0))

    np.save(save_path, np.median(percentiles, axis=0))