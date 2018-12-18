import parser
import graph_utils
import tensor_utils
import segmentation_scoring
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import demo_utils
from glob import glob


def makedir_if_needed(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    args = parser.parse_args()

    output_tensor_dir = os.path.join(args.output_dir, args.output_node.replace("/", '_'))
    makedir_if_needed(output_tensor_dir)

    print("Loading graph\n")
    graph = graph_utils.load_graph(args.graph)

    print("Processing images tensors\n")
    graph_utils.process_image_tensors(
        graph=graph,
        input_node_name=args.input_node,
        output_node_name=args.output_node,
        dataset_dir=os.path.join(args.dataset_dir, "images"),
        save_dir=output_tensor_dir
    )

    print("Calculating percentiles for each filter\n")
    percentile_path = os.path.join(
        args.output_dir,
        "_".join(["percentile", str(args.percentile), args.output_node.replace('/', '_'), ".npy"])
    )
    tensor_utils.calculate_percentiles(
        tensor_dir=output_tensor_dir,
        save_path=percentile_path,
        percentile=args.percentile
    )

    print("Creating result dataframe\n")
    labels_pd = pd.read_csv(os.path.join(args.dataset_dir, "labels.csv"), sep=',')
    labels_pd.index = labels_pd['number']
    result_pd = segmentation_scoring.create_result_dataframe(args.n_maps, labels_pd)
    source_filenames = os.listdir(os.path.join(args.dataset_dir, "images"))
    percentiles = np.load(percentile_path)

    for source_filename in tqdm(source_filenames):
        source_image_path = os.path.join(args.dataset_dir, 'images', source_filename)
        label_image_path = os.path.join(args.dataset_dir, 'labeling', source_filename)
        labeled_image = segmentation_scoring.imread_int_labels(label_image_path)
        img_labels_set = set(labeled_image.flatten()).intersection(labels_pd.index)

        tensor_path = os.path.join(output_tensor_dir, source_filename + ".npy")
        tensor = np.load(tensor_path)

        for label in img_labels_set:
            bin_mask_labels = labeled_image == label
            for i in range(args.n_maps):
                bin_preds = segmentation_scoring.preprocess_map(tensor[:, :, i], percentiles[i])
                intersection, union = segmentation_scoring.compute_bin_iou(bin_preds, bin_mask_labels)

                result_pd.at[(label, i), 'intersection'] += intersection
                result_pd.at[(label, i), 'union'] += union
                result_pd.at[(label, i), 'num_images'] += 1
                iou_score = intersection * 1. / union
                if iou_score > result_pd.at[(label, i), 'best_score']:
                    result_pd.at[(label, i), 'best_score'] = iou_score
                    result_pd.at[(label, i), 'best_image'] = source_image_path
                    result_pd.at[(label, i), 'best_tensor'] = tensor_path
                    result_pd.at[(label, i), 'best_label_img'] = label_image_path

    result_pd = result_pd[result_pd['union'] != 0]
    result_pd['iou'] = result_pd['intersection'] / result_pd['union']
    result_pd = result_pd.sort_values('iou', ascending=False)

    output_name = "output_scores"
    result_pd.to_csv(os.path.join(args.output_dir, output_name + ".csv"))
    makedir_if_needed(os.path.join(args.output_dir, "demo"))

    print("Making demo: " + os.path.join(args.output_dir, "demo"))
    demo_utils.make_demo(result_pd, os.path.join(args.output_dir, "demo"), percentiles, top_k=10)


if __name__ == "__main__":
    main()