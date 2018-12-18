import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
import skimage
import skimage.io


def load_graph(frozen_graph_filename):
    print("Loading: " + frozen_graph_filename)
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def load_batch(filenames):
    batch = map(lambda x: skimage.img_as_float(skimage.io.imread(x)), filenames)
    batch = np.array(list(batch))
    return batch


def save_batch(filenames, tensors, folder):
    for filename, tensor in zip(filenames, tensors):
        np.save(os.path.join(folder, filename.replace('/', '_') + '.npy'), tensor)


def iterate_minibatches(all_filenames, batch_size, images_folder):
    for i in tqdm(range(0, len(all_filenames), batch_size)):
        batch_filenames = all_filenames[i:i+batch_size]
        batch_filenames_full = list(map(lambda x: os.path.join(images_folder, x), batch_filenames))
        yield batch_filenames, load_batch(batch_filenames_full)


def process_image_tensors(graph, input_node_name, output_node_name, dataset_dir, save_dir, batch_size=32):
    input_node = graph.get_tensor_by_name(input_node_name + ":0")
    output_node = graph.get_tensor_by_name(output_node_name + ":0")
    sess = tf.Session(graph=graph)

    image_filenames = os.listdir(dataset_dir)
    for batch_filenames, batch_tensors in iterate_minibatches(image_filenames, batch_size, dataset_dir):
        output_tensors = sess.run([output_node], feed_dict={input_node: batch_tensors})[0]
        save_batch(batch_filenames, output_tensors, save_dir)