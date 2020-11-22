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
from MaXplore.configs import args_3


def exp_3():
    tf.compat.v1.enable_eager_execution()
    
    # Multi-Topk-MaXplore
    if(os.path.isdir('./output/gen_test_exp3/')):
        files = glob.glob('./output/gen_test_exp3/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir('./output/gen_test_exp3/')

    # direct the output the txt file
    sys.stdout = open("./output/max_stat_exp3.txt", "w")

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
    model2 = Model2(input_tensor=input_tensor)
    model3 = Model3(input_tensor=input_tensor)

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

    conv1_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('block1_conv1').output)
    conv2_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('block2_conv1').output)
    model1_layer_model = Model(inputs=model1.inputs, outputs=model1.get_layer('before_softmax').output)
    model2_layer_model = Model(inputs=model2.inputs, outputs=model2.get_layer('before_softmax').output)
    model3_layer_model = Model(inputs=model3.inputs, outputs=model3.get_layer('before_softmax').output)
    gen_input_num = 0
    ratio_dict_list = []

    for e in tqdm(range(args_3.Data_Num)):
        ratio_dict_temp_list = []
        print('\n\n\n', '-' * 10, 'Iter', e, '-' * 10)

        # random select a test image
        data_index = random.choice(list(range(x_test.shape[0])))
        gen_img = np.expand_dims(x_test[data_index, :, :], axis=0)
        orig_img = gen_img.copy()

        # first check if input already induces differences
        label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
            model3.predict(gen_img)[0])
        orig_label = label1

        if(label1 == label2 == label3):
            # get the filter weight nuerons
            output_indices_dict = get_output_top_neurons(gen_img, model1, top=args_3.Topk)

            # update max dict for the test image
            update_test_max_stat(model1, gen_img, model_max_dict1)

            # define prediction on intermediate output
            layer_names, intermediate_layer_model = get_layer_model(model1)
            
            for layer, outputs_list in output_indices_dict.items():
                for filter_num, output_indices in enumerate(outputs_list):
                    if(args_3.one_filter):
                        if(layer != 'block1_conv1' or filter_num > 0):
                            continue
                    print('#' * 10, 'Layer:', layer, 'Filter:', filter_num, '#' * 10)

                    # reset to original images
                    gen_img = orig_img.copy()
                    gen_img = tf.convert_to_tensor(gen_img)

                    for step in range(1, args_3.Iterations + 1):
                        with tf.GradientTape() as tape:
                            tape.watch(gen_img)
                            if(layer == 'block1_conv1'):
                                pred = conv1_layer_model(gen_img)
                            else:
                                pred = conv2_layer_model(gen_img)

                            # keep track of all nuerons loss
                            loss_neurons = None
                            for k in range(args_3.Topk):
                                index2, index3 = tuple(output_indices[k, :])
                                if(loss_neurons):
                                    loss_neurons += pred[0, index2, index3, filter_num]
                                else:
                                    loss_neurons = pred[0, index2, index3, filter_num]

                            # loss from model difference
                            loss_1 = -args_3.weight_diff * model1_layer_model(gen_img)[..., orig_label]
                            loss_2 = model2_layer_model(gen_img)[..., orig_label]
                            loss_3 = model3_layer_model(gen_img)[..., orig_label]
                            loss_model = loss_1 + loss_2 + loss_3
                            final_loss = loss_model + args_3.weight_nc * loss_neurons

                            # compute gradients
                            grad_values = tape.gradient(final_loss, [gen_img])

                            # grad_values = tf.math.add_n(grad_values)
                        gen_img += grad_values[0] * args_3.Stepsize

                        # check new predictions
                        predictions1 = np.argmax(model1.predict(gen_img)[0])
                        predictions2 = np.argmax(model2.predict(gen_img)[0])
                        predictions3 = np.argmax(model3.predict(gen_img)[0])

                        if not predictions1 == predictions2 == predictions3:
                            ratio_dict = get_layer_max_stat(model1, gen_img, model_max_dict1, None, layer_names, intermediate_layer_model)
                            print(layer, filter_num)
                            print(ratio_dict)
                            ratio_dict_temp_list.append(ratio_dict)
                            gen_img = tf.reshape(gen_img, [gen_img.shape[1], gen_img.shape[2], 1])
                            tf.keras.preprocessing.image.save_img(
                                f'./output/gen_test_exp3/Data{e}_{layer}_{filter_num}_{predictions1}_{predictions2}_{predictions3}.png', gen_img.numpy())
                            gen_input_num += 1
                            break
                    sys.stdout.flush()
        save_img = tf.reshape(orig_img, [orig_img.shape[1], orig_img.shape[2], 1])
        tf.keras.preprocessing.image.save_img(
                    f'./output/gen_test_exp3/Data{e}_orig.png', save_img.numpy())

        d = {}
        for k in layer_names[1:]:
            d[k] = 0
            if(len(ratio_dict_temp_list) > 0):
                for item in ratio_dict_temp_list:
                    d[k] += item[k]
                d[k] = d[k] / len(ratio_dict_temp_list)
        if(len(ratio_dict_temp_list) > 0):
            ratio_dict_list.append(d)

    # plot   
    # Multi-Topk-MaXplore dataframe
    plot_dict = {x: [] for x in ratio_dict_list[0].keys()}
    for item in ratio_dict_list:
        for k in item.keys():
            plot_dict[k].append(item[k])
    df_exp3 = pd.melt(pd.DataFrame(plot_dict))
    df_exp3.columns = ['Layer', 'Layer-SNACov']
    df_exp3['Method'] = 'Multi-Topk-MaXplore'

    # DeepXplore dataframe
    import json
    with open('./output/deepxplore.txt', 'r') as f:
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
    df_deep = pd.melt(pd.DataFrame(plot_dict))
    df_deep.columns = ['Layer', 'Layer-SNACov']
    df_deep['Method'] = 'DeepXplore'

    plt.figure()
    df = pd.concat([df_exp3, df_deep], ignore_index=True)
    img = sns.boxplot(x="Layer", y="Layer-SNACov", hue='Method', data=df)
    img.tick_params(labelsize=10)
    img.set_xlabel("Layer",fontsize=15)
    img.set_ylabel("Layer-SNACov",fontsize=15)
    img.get_figure().savefig("./output/exp3_plot.jpg")





