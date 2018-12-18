import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import segmentation_scoring
import numpy as np
from skimage.io import imread
from skimage import transform
import segmentation_scoring
from glob import glob


def write_html(folder, html_path):
    imgs = glob(os.path.join(folder, "*.png"))
    with open(html_path, "a") as f:
        f.write("Some examples of segmentation in format <br>\n")
        f.write("Source image - segmentation labeling - features map's highest activations with iou score (in %, * 100) <br>\n")
        for imgpath in imgs:
            f.write("<img src=\"./{}\"> <br>\n".format(os.path.basename(imgpath)))


def imwrite_top_k_results(scores, output_folder, percentiles, top_k=2, by='best_score'):
    scores = scores.sort_values(by, ascending=False)

    for number in range(top_k):
        best_example = scores.ix[number]
        img = imread(best_example["best_image"])
        tensor = np.load(best_example['best_tensor'])
        best_map = scores.index[number][1]
        best_label = scores.index[number][0]
        labeling = segmentation_scoring.imread_int_labels(best_example['best_label_img']) == best_label
        f, axs = plt.subplots(1,3, figsize=(8, 24))
        axs[0].set_title("image")
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[1].set_title(best_example['label_name'] )
        axs[1].imshow(labeling, cmap='gray')
        axs[1].axis('off')
        axs[2].set_title("IOU: " + str(int(np.round(best_example['best_score'] * 100))) + "%")
        axs[2].imshow(segmentation_scoring.preprocess_map(
            tensor[:, :, best_map], percentiles[best_map])[:, :, np.newaxis] * transform.resize(img, (113, 113)))
        axs[2].axis('off')
        plt.savefig(os.path.join(output_folder, str(number) + ".png"), bbox_inches='tight')

    write_html(output_folder, os.path.join(output_folder, "demo.html"))


def make_demo(result_pd, output_folder, percentiles, top_k=2, by='best_score'):
    scores = result_pd[result_pd['union'] != 0]
    scores['iou'] = scores['intersection'] / scores['union']
    scores['is_detector'] = scores['iou'] > 0.04

    with open(os.path.join(output_folder, "demo.html"), 'wa') as f:
        f.write("Most detectable objects in format label - fraction of unique detectors <br>\n")
        f.write("(Feature map is unique detector for object if mean intersection over union > 0.04):<br>\n")
        mean_scores = scores.groupby('label_name')['is_detector'].mean().sort_values(ascending=False).head(5)
        for label, score in mean_scores.iteritems():
            f.write("{}: {}<br>\n".format(label, score))

    imwrite_top_k_results(scores, output_folder, percentiles, top_k=10, by='best_score')

