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

keras_idx_to_class = {
    0: 1, 1: 10, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106,
    9: 107, 10: 108, 11: 109, 12: 11, 13: 110, 14: 111, 15: 112, 16: 113,
    17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 23: 12, 24: 120,
    25: 121, 26: 122, 27: 123, 28: 124, 29: 125, 30: 126, 31: 127, 32: 128,
    33: 129, 34: 13, 35: 130, 36: 131, 37: 132, 38: 133, 39: 134, 40: 135,
    41: 136, 42: 137, 43: 138, 44: 139, 45: 14, 46: 140, 47: 141, 48: 142,
    49: 143, 50: 144, 51: 145, 52: 146, 53: 147, 54: 148, 55: 149, 56: 15,
    57: 150, 58: 151, 59: 152, 60: 153, 61: 154, 62: 155, 63: 156, 64: 157,
    65: 158, 66: 159, 67: 16, 68: 160, 69: 161, 70: 162, 71: 163, 72: 164,
    73: 165, 74: 166, 75: 167, 76: 168, 77: 169, 78: 17, 79: 170, 80: 171,
    81: 172, 82: 173, 83: 174, 84: 175, 85: 176, 86: 177, 87: 178, 88: 179,
    89: 18, 90: 180, 91: 181, 92: 182, 93: 183, 94: 184, 95: 185, 96: 186,
    97: 187, 98: 188, 99: 189, 100: 19, 101: 190, 102: 191, 103: 192,
    104: 193, 105: 194, 106: 195, 107: 196, 108: 197, 109: 198, 110: 199,
    111: 2, 112: 20, 113: 200, 114: 201, 115: 202, 116: 203, 117: 204,
    118: 205, 119: 206, 120: 207, 121: 208, 122: 209, 123: 21, 124: 210,
    125: 211, 126: 212, 127: 213, 128: 214, 129: 215, 130: 216, 131: 217,
    132: 218, 133: 219, 134: 22, 135: 220, 136: 221, 137: 222, 138: 223,
    139: 224, 140: 225, 141: 226, 142: 227, 143: 228, 144: 229, 145: 23,
    146: 230, 147: 231, 148: 232, 149: 233, 150: 234, 151: 235, 152: 236,
    153: 237, 154: 238, 155: 239, 156: 24, 157: 240, 158: 241, 159: 242,
    160: 243, 161: 244, 162: 245, 163: 246, 164: 247, 165: 248, 166: 249,
    167: 25, 168: 250, 169: 251, 170: 252, 171: 253, 172: 254, 173: 255,
    174: 256, 175: 26, 176: 27, 177: 28, 178: 29, 179: 3, 180: 30, 181: 31,
    182: 32, 183: 33, 184: 34, 185: 35, 186: 36, 187: 37, 188: 38, 189: 39,
    190: 4, 191: 40, 192: 41, 193: 42, 194: 43, 195: 44, 196: 45, 197: 46,
    198: 47, 199: 48, 200: 49, 201: 5, 202: 50, 203: 51, 204: 52, 205: 53,
    206: 54, 207: 55, 208: 56, 209: 57, 210: 58, 211: 59, 212: 6, 213: 60,
    214: 61, 215: 62, 216: 63, 217: 64, 218: 65, 219: 66, 220: 67, 221: 68,
    222: 69, 223: 7, 224: 70, 225: 71, 226: 72, 227: 73, 228: 74, 229: 75,
    230: 76, 231: 77, 232: 78, 233: 79, 234: 8, 235: 80, 236: 81, 237: 82,
    238: 83, 239: 84, 240: 85, 241: 86, 242: 87, 243: 88, 244: 89, 245: 9,
    246: 90, 247: 91, 248: 92, 249: 93, 250: 94, 251: 95, 252: 96, 253: 97,
    254: 98, 255: 99
}

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
                 # layer.input_shape[1] * layer.input_shape[2] *      # picture
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
        checked_sizes = [224, 227, 128, 64, 32]
        resized = True
        for size in checked_sizes:
            try:
                model.predict(np.random.randn(1, size, size, 3))
                break
            except ValueError:
                pass

    start = time.time()

    correct_zero_based = 0
    corrert_one_based = 0
    correct_keras_based = 0
    for idx, row in public_images.iterrows():
        image_path = os.path.join(images_dir, row['image'])
        image = imread(image_path)
        if len(image.shape) != 3:
            image = gray_to_color(image)
        if resized:
            image = imresize(image, (size, size, 3))

        image = np.expand_dims(image, axis=0) / 255
        true_class_one_based = row['class']
        true_class_zero_based = true_class_one_based - 1
        pred_class = np.argmax(model.predict(image)[0])
        if pred_class == true_class_zero_based:
            correct_zero_based += 1
        if pred_class == true_class_one_based:
            corrert_one_based += 1
        if keras_idx_to_class[pred_class] == true_class_one_based:
            correct_keras_based += 1

    time_cons = time.time() - start
    total_images = public_images.shape[0]
    accuracy_zero_based = correct_zero_based / total_images
    accuracy_one_based = corrert_one_based / total_images
    accuracy_keras_based = correct_keras_based / total_images
    best_acc = max(accuracy_zero_based, accuracy_one_based, accuracy_keras_based)
    return best_acc, time_cons


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
