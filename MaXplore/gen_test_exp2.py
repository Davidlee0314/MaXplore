import argparse
import pickle as pkl
import os 
import random
random.seed(7)
import time
import copy
import glob
import sys
import logging
sys.path.append("..")

from keras.datasets import mnist
from keras.layers import Input
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tf.random.set_seed(7)


from Models.Model1 import Model1
from Models.Model2 import Model2
from Models.Model3 import Model3
from MaXplore.utils_max import *
from MaXplore.configs import args_2

def exp_2():
    tf.compat.v1.enable_eager_execution()
    
    if(os.path.isdir('./output/gen_test_exp2/')):
        files = glob.glob('./output/gen_test_exp2/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir('./output/gen_test_exp2/')

    # direct the output the txt file
    sys.stdout = open("./output/max_stat_exp2.txt", "w")

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    # load multiple models sharing same input tensor
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

    ratio_dict_list = []

    # define prediction on intermediate output
    layer_names, intermediate_layer_model = get_layer_model(model1)
    conv1_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('block1_conv1').output)
    conv2_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('block2_conv1').output)
    model1_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('before_softmax').output)

    for e in tqdm(range(args_2.Data_Num)):
        print('\n\n\n', '-' * 10, 'Iter', e, '-' * 10)

        # random select a test image
        data_index = random.choice(list(range(x_test.shape[0])))
        gen_img = np.expand_dims(x_test[data_index, :, :], axis=0)
        orig_img = gen_img.copy()
        orig_label = np.argmax(model1(gen_img))

        # get the filter weight nuerons
        output_indices_dict = get_output_top_neurons(gen_img, model1, top=args_2.Topk)

        # update max dict for the test image
        update_test_max_stat(model1, gen_img, model_max_dict1)
        for layer, outputs_list in output_indices_dict.items():
            if(layer != 'block1_conv1'):
                continue
            for filter_num, output_indices in enumerate(outputs_list):
                print('#' * 10, 'Layer:', layer, 'Filter:', filter_num, '#' * 10)

                # reset to original images
                gen_img = orig_img.copy()
                gen_img = tf.convert_to_tensor(gen_img)

                for step in range(1, args_2.Iterations + 1):
                    with tf.GradientTape() as tape:
                        tape.watch(gen_img)
                        pred = conv1_layer_model(gen_img)

                        # keep track of all nuerons loss
                        loss_neurons = None
                        for k in range(args_2.Topk):
                            index2, index3 = tuple(output_indices[k, :])
                            if(loss_neurons):
                                loss_neurons += pred[0, index2, index3, filter_num]
                            else:
                                loss_neurons = pred[0, index2, index3, filter_num]

                        # loss from model difference
                        final_loss = loss_neurons

                        # compute gradients
                        grad_values = tape.gradient(final_loss, [gen_img])

                        # grad_values = tf.math.add_n(grad_values)
                    gen_img += grad_values[0] * args_2.Stepsize

                ratio_dict = get_layer_max_stat(model1, gen_img, model_max_dict1, None, layer_names, intermediate_layer_model)
                print(ratio_dict)
                ratio_dict_list.append(ratio_dict)
                gen_img = tf.reshape(gen_img, [gen_img.shape[1], gen_img.shape[2], 1])
                tf.keras.preprocessing.image.save_img(
                    f'./output/gen_test_exp2/Data{e}_{layer}_{filter_num}.png', gen_img.numpy())
        save_img = tf.reshape(orig_img, [orig_img.shape[1], orig_img.shape[2], 1])
        tf.keras.preprocessing.image.save_img(
                    f'./output/gen_test_exp2/Data{e}_orig.png', save_img.numpy())

    # plot the result
    # Topk-MaXplore dataframe
    plot_dict = {x: [] for x in ratio_dict_list[0].keys()}
    for item in ratio_dict_list:
        for k in item.keys():
            plot_dict[k].append(item[k])
    df_exp2 = pd.melt(pd.DataFrame(plot_dict))
    df_exp2.columns = ['Layer', 'Layer-SNACov']
    df_exp2['Method'] = 'Topk-MaXplore'

    # One-MaXplore dataframe
    import json
    with open('./output/max_stat_exp1.txt', 'r') as f:
        txt = f.read()
        temp_li = txt.split('\n')
        ratio_dict_list = []
        for i in temp_li:
            if('{' in i):
                i = i.replace("'", "\"")
                index = i.find('}')
                ratio_dict_list.append(json.loads(i))
    plot_dict = {x: [] for x in ratio_dict_list[0].keys()}
    for item in ratio_dict_list:
        for k in item.keys():
            plot_dict[k].append(item[k])
    df_exp1 = pd.melt(pd.DataFrame(plot_dict))
    df_exp1.columns = ['Layer', 'Layer-SNACov']
    df_exp1['Method'] = 'One-MaXplore'

    plt.figure()
    df = pd.concat([df_exp1, df_exp2], ignore_index=True)
    img = sns.boxplot(x="Layer", y="Layer-SNACov", hue='Method', data=df)
    img.tick_params(labelsize=10)
    img.set_xlabel("Layer",fontsize=15)
    img.set_ylabel("Layer-SNACov",fontsize=15)
    img.get_figure().savefig("./output/exp2_plot.jpg")
