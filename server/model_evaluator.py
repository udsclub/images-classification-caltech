import time
import random

import redis
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from keras import backend as K
import numpy as np


redis_db = redis.StrictRedis(
    host='localhost', port=6379, db=1,
    charset="utf-8", decode_responses=True)
results_db = redis.StrictRedis(
    host='localhost', port=6379, db=2,
    charset="utf-8", decode_responses=True)


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
    # TODO
    return random.randint(1, 100)


def get_all_results(model_path):
    model = load_model_my(model_path)
    results = {}
    try:
        results['operations'] = count_operations(model)
    except Exception:
        results['operations'] = 0
    try:
        results['model_size'] = get_model_size(model)
    except Exception:
        results['model_size'] = 0
    try:
        results['accuracy'] = get_model_accuracy(model)
    except Exception:
        results['accuracy'] = 0
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
