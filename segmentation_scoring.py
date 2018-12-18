import pandas as pd
from skimage import transform, io
import numpy as np

def create_result_dataframe(num_maps, labels_pd):
    index = [(ind, imap) for ind in labels_pd.index for imap in range(num_maps)]
    label_name = [labels_pd['name'][x[0]] for x in index]
    result_pd = pd.DataFrame({
        '(label, activation_map)': index,
        'label_name': label_name,
        'intersection': 0.,
        'union': 0.,
        'iou': 0.,
        'num_images': 0,
        'best_image': None,
        'best_score': 0.,
        'best_tensor': None,
        'best_label_img': None
    })
    result_pd.index = result_pd['(label, activation_map)']
    del result_pd['(label, activation_map)']
    return result_pd

def preprocess_map(preds, threshold, shape=(113, 113)):
    resized = transform.resize(preds, shape)
    return resized > threshold


def compute_bin_iou(preds, mask):
    intersection = np.sum(preds * mask, dtype=np.float) * 1.
    union = np.sum(preds + mask > 0) * 1.
    return intersection, union


def imread_int_labels(mask_path):
    image = io.imread(mask_path)
    return np.array((image[:, :, 0] + image[:, :, 1] * 256), dtype=np.int32)
