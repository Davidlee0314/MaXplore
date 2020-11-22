import argparse
import pickle as pkl
import os 
import random
random.seed(7)
import time
import copy
import glob
import sys

from keras.datasets import mnist
from keras.layers import Input
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
tf.random.set_seed(7)

from Models.Model1 import Model1
from MaXplore.utils_max import *
from MaXplore.configs import args_1


def exp_1():
    tf.compat.v1.enable_eager_execution()
    
    if(os.path.isdir('./output/gen_test_exp1/')):
        files = glob.glob('./output/gen_test_exp1/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir('./output/gen_test_exp1/')

    # direct the output the txt file
    sys.stdout = open("./output/max_stat_exp1.txt", "w")

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # the data, shuffled and split between train and test sets
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    # load model sharing same input tensor
    model1 = Model1(input_tensor=input_tensor)

    if(os.path.exists('./data/model1_max_dict.pkl')):
        with open('./data/model1_max_dict.pkl', 'rb') as f:
            model_max_dict1 = pkl.load(f)
            for key in model_max_dict1:
                model_max_dict1[key] = model_max_dict1[key].numpy()
    else:
        # init max table to store statistics of nuerons
        model_max_dict1 = init_max_tables(model1)
        # track neurons' maximum from x_train
        track_max_stat(model1, x_train, model_max_dict1)
        with open('./data/model1_max_dict.pkl', 'wb') as f:
            pkl.dump(model_max_dict1, f)

    # random select a test image that yield no difference on three models
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_label = np.argmax(model1(gen_img))
    orig_img = gen_img.copy()

    # update max dict for the test image
    update_test_max_stat(model1, gen_img, model_max_dict1)

    # loop over first layer neuron
    layer_name1 = 'block1_conv1'
    change_max_count = 0
    change_shallow_max_count = 0
    ratio_dict_list = []

    # define prediction on intermediate output
    layer_names, intermediate_layer_model = get_layer_model(model1)
    conv1_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer(layer_name1).output)

    for i, j, k in tqdm(np.ndindex(tuple(model_max_dict1[layer_name1].shape))):
        # reset to original images
        gen_img = orig_img.copy()
        gen_img = tf.convert_to_tensor(gen_img)

        for step in range(1, args_1.Iterations + 1):
            with tf.GradientTape() as tape:
                tape.watch(gen_img)
                pred = conv1_layer_model(gen_img)
                loss = pred[..., i, j, k]
                grad_value = tape.gradient(loss, [gen_img])[0]
            gen_img += grad_value * args_1.Stepsize

        neuron_output, old_neuron_max, ratio_dict = get_layer_max_stat(model1, gen_img, model_max_dict1, (i, j, k), layer_names, intermediate_layer_model)
        if(neuron_output > old_neuron_max):
            print('Neuron:', i, j, k, '\tnew value:', neuron_output, '\told value', old_neuron_max)
            change_max_count += 1
            for key, value in ratio_dict.items():
                if(value > 0):
                    change_shallow_max_count += 1
                    break
            print(ratio_dict)
            ratio_dict_list.append(ratio_dict)
            if(np.sum(gen_img.numpy() != orig_img) > 0):
                save_img = tf.reshape(gen_img, [gen_img.shape[1], gen_img.shape[2], 1])
                tf.keras.preprocessing.image.save_img(
                    f'./output/gen_test_exp1/Neuron({i},{j},{k})_Label({orig_label}).png', save_img.numpy())

    # save original image for comparison
    save_img = tf.reshape(orig_img, [orig_img.shape[1], orig_img.shape[2], 1])
    tf.keras.preprocessing.image.save_img(
                    f'./output/gen_test_exp1/orig.png', save_img.numpy())
    print(f'Max stat change: {change_max_count}\t Max shallow stat change: {change_shallow_max_count}\tChange ratio: {change_shallow_max_count / change_max_count}')
            
    # plot boxplot for each layer 
    plot_dict = {x: [] for x in ratio_dict_list[0].keys()}
    for item in ratio_dict_list:
        for k in item.keys():
            plot_dict[k].append(item[k])

    plt.figure()
    df = pd.melt(pd.DataFrame(plot_dict))
    df.columns = ['Layer', 'Layer-SNACov']
    img = sns.boxplot(x="Layer", y="Layer-SNACov", data=df)
    img.tick_params(labelsize=10)
    img.set_xlabel("Layer",fontsize=15)
    img.set_ylabel("Layer-SNACov",fontsize=15)
    img.get_figure().savefig("./output/exp1_plot.jpg")
