from collections import defaultdict
import random
import time

from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np


def normalize(x):
    '''
    Utility function to normalize a tensor by its L2 norm
    '''
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def constraint_light(gradients):
    '''
    Utility function to constraint gradients into brightness condition
    '''
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads

def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (random.randint(0, gradients.shape[1] - rect_shape[0]), 
        random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def init_max_tables(model):
    '''
    Initialize dictionary for storing maximum values of neurons
    '''
    model_max_dict = defaultdict(float)
    init_dict(model, model_max_dict)
    return model_max_dict

def init_dict(model, model_max_dict):
    '''
    Initialize dictionary for storing maximum values of neurons
    '''
    for layer in model.layers:
        # ignore input, flatten, prediction nueron max statistics
        if 'flatten' in layer.name or 'input' in layer.name or 'predictions' in layer.name:
            continue
        # init max dict to 0
        model_max_dict[layer.name] = tf.zeros(layer.output_shape[1:])

def track_max_stat(model, x_train, model_max_dict):
    '''
    Track the maximum values for training set
    '''
    layer_names, intermediate_layer_model = get_layer_model(model)
    # track statistics with x_train input
    for x_index in range(x_train.shape[0]):
        x = np.expand_dims(x_train[x_index, ...], axis=0)
        intermediate_layer_outputs = intermediate_layer_model.predict(x)
        
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            model_max_dict[layer_names[i]] = tf.math.maximum(
                model_max_dict[layer_names[i]], intermediate_layer_output[0])

def update_test_max_stat(model, x_test, model_max_dict):
    '''
    Track the maximum values for a single testing data
    '''
    layer_names, intermediate_layer_model = get_layer_model(model)
    intermediate_layer_outputs = intermediate_layer_model.predict(x_test)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        if(tf.executing_eagerly()):
            model_max_dict[layer_names[i]] = np.maximum(
                model_max_dict[layer_names[i]], intermediate_layer_output[0])
        else:
            model_max_dict[layer_names[i]] = tf.math.maximum(
                model_max_dict[layer_names[i]], intermediate_layer_output[0])

def get_layer_model(model):
    '''
    Get layer names and initialize layer model to get layers' outputs
    '''
    layer_names = [layer.name for layer in model.layers 
        if 'flatten' not in layer.name and 'input' not in layer.name and 'predictions' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    return layer_names, intermediate_layer_model

def get_layer_max_stat(model, x_test, model_max_dict, target_neuron_index, layer_names, intermediate_layer_model):
    '''
    Update the max stat and return two information
    1. What the new output value and the maximum value for the target_neuron_index
    2. How many nueurons from shallower layers also change the maximum value
    '''
    # layer_names, intermediate_layer_model = get_layer_model(model)
    intermediate_layer_outputs = intermediate_layer_model.predict(x_test)
    
    # for the first task 
    if(target_neuron_index):
        neuron_output = intermediate_layer_outputs[0][0, target_neuron_index[0], target_neuron_index[1], target_neuron_index[2]]
        old_neuron_max = model_max_dict[layer_names[0]][target_neuron_index[0], target_neuron_index[1], target_neuron_index[2]]
    
    # for the second task
    total_count = 0
    update_count = 0
    ratio_dict = {}
    
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        bool_arr = intermediate_layer_output[0] > model_max_dict[layer_names[i]]
        if(tf.executing_eagerly()):
            update_count = np.count_nonzero(bool_arr)
            total_count = bool_arr.reshape(-1, 1).shape[0]
        else:
            update_count = np.count_nonzero(bool_arr.eval(session=tf.compat.v1.Session()))
            total_count = bool_arr.eval(session=tf.compat.v1.Session()).reshape(-1, 1).shape[0]
        
        if(i != 0):
            ratio_dict[layer_names[i]] = update_count / total_count
        update_count = 0 
        total_count = 0
    
    if(target_neuron_index):
        return neuron_output, old_neuron_max, ratio_dict
    else:
        return ratio_dict


# Experiment 2 

def get_output_top_neurons(x_test, model, top):
    '''
    Get top k neurons in a filter to be included in the gradient descent
    Return: dict with list length equal to filter maps
    '''
    output_indices_dict = {}
    for layer in model.layers:
        if('conv' in layer.name):
            output_indices_dict[layer.name] = []
            layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer.name).output)
            outputs = layer_model(x_test)
            for w in range(outputs.shape[-1]):
                _, indices = tf.math.top_k(tf.reshape(outputs[..., w], [-1]), k=top)
                indices = np.array(np.unravel_index(indices.numpy(), outputs[..., w].shape)).T
                output_indices_dict[layer.name].append(indices[:, 1:])
    return output_indices_dict