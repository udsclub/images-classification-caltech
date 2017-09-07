import os
import time
import random

import redis
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
from tqdm import tqdm


redis_db = redis.StrictRedis(
    host='localhost', port=6379, db=1,
    charset="utf-8", decode_responses=True)
results_db = redis.StrictRedis(
    host='localhost', port=6379, db=2,
    charset="utf-8", decode_responses=True)


### Some settings
# base_dir = '/Users/illarionkhliestov/Datasets/caltech-256/'
base_dir = '/home/ubuntu/Datasets/caltech-256'
csv_path = os.path.join(base_dir, 'kaggle_test.csv')
df = pd.read_csv(csv_path)
images_dir = os.path.join(base_dir, 'test')
public_images = df[df['Usage'] == 'Public']


def gray_to_color(image):
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)


def load_model_my(model_path):
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        model = load_model(model_path)
    return model


# Count operation in loaded model
def count_operations(model):
    ops = 0
    for layer in model.layers:
        w = 0
        if 'units' in layer.get_config():
            w = layer.input_shape[1] * layer.output_shape[1]
        if 'kernel_size' in layer.get_config():
            w = (layer.kernel_size[0] * layer.kernel_size[1] *
                 layer.input_shape[1] * layer.input_shape[2] *      # picture
                 layer.input_shape[3] * layer.output_shape[3] //    # channels
                 layer.strides[0] // layer.strides[1])
            if 'conv_dw_' in layer.name:
                w //= layer.output_shape[3]
        ops += w
    return ops


def get_model_size(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count + non_trainable_count


def get_model_accuracy(model):
    # get first predition to choose resolution of the images
    try:
        model.predict(np.random.randn(1, 374, 530, 3))
        resized = False
    except ValueError:
        resized = True

    start = time.time()

    correct = 0
    for idx, row in public_images.iterrows():
        image_path = os.path.join(images_dir, row['image'])
        image = imread(image_path)
        if len(image.shape) != 3:
            image = gray_to_color(image)
        if resized:
            image = imresize(image, (224, 224, 3))

        image = np.expand_dims(image, axis=0) / 255
        true_class = row['class'] - 1
        pred_class = np.argmax(model.predict(image)[0])
        if pred_class == true_class:
            correct += 1

    time_cons = time.time() - start
    accuracy = correct / public_images.shape[0]
    return accuracy, time_cons


def get_all_results(model_path):
    results = {
        'operations': 0,
        'model_size': 0,
        'accuracy': 0,
        'time_cons': 0,
    }
    try:
        model = load_model_my(model_path)
    except Exception as e:
        return results
    try:
        results['operations'] = count_operations(model)
    except Exception:
        pass
    try:
        results['model_size'] = get_model_size(model)
    except Exception:
        pass
    try:
        results['accuracy'], results['time_cons'] = get_model_accuracy(model)
    except Exception:
        pass
    del model
    return results


if __name__ == '__main__':
    while True:
        task = redis_db.lpop('scheduled_tasks')
        if not task:
            time.sleep(1)
            continue
        print("handle task:", task)
        model_results = get_all_results(task)
        results_db.hmset(task, model_results)
