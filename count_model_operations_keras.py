from keras.models import model_from_json
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import os

# Use this function to save your model to .json file
def save_model_json(model, filename):
    with open(filename, 'w') as fOut:
        fOut.write(model.to_json())
        fOut.closed

# Load model from .json file
def get_model(model_file):
    with open(model_file, 'r') as fIn:
        data = fIn.read()
        with CustomObjectScope({'relu6':relu6, 'DepthwiseConv2D':DepthwiseConv2D}):
            model = model_from_json(data)
        fIn.closed
        
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
        #print(layer.name, w, layer.input_shape , layer.output_shape, layer.get_config(), '\n')
    return ops   

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    parser = ArgumentParser(description="", formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--file', dest='file', action='store', default="", help='path/your_model.json')

    args = parser.parse_args()
    
    filename = args.file
    if len(filename) > 0 and filename.endswith('.json') :
        model = get_model(filename)
        print('Operations:', count_operations(model))
        print('Parameters:', model.count_params())
    else:
        print('Please specify .json file to process. Run --help for more info.')
        exit(0)
