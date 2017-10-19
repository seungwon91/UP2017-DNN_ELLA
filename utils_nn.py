import numpy as np
import tensorflow as tf
from math import ceil

from utils import *

############################################################
#####   functions for adding fully-connected network   #####
############################################################
#### function to add fully-connected layer
def new_fc_layer(layer_input, input_dim, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None):
    with tf.name_scope('fc_layer'):
        if weight is None:
            weight = new_weight(shape=[input_dim, output_dim])
        if bias is None:
            bias = new_bias(shape=[output_dim])

        if activation_fn is None:
            layer = tf.matmul(layer_input, weight) + bias
        elif activation_fn is 'classification':
            #layer = tf.nn.softmax(tf.matmul(layer_input, weight) + bias)
            layer = tf.matmul(layer_input, weight) + bias
        else:
            layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

#### function to generate network of fully-connected layers
####      'dim_layers' contains input/output layer
def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None, tensorboard_name_scope='fc_net'):
    if len(dim_layers) < 2:
        #### for the case that hard-parameter shared network does not have shared layers
        return (net_input, [])
    elif params is None:
        with tf.name_scope(tensorboard_name_scope):
            layers, params = [], []
            for cnt in range(len(dim_layers)-1):
                if cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(net_input, dim_layers[0], dim_layers[1], activation_fn=activation_fn)
                elif cnt == len(dim_layers)-2 and output_type is 'classification':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn='classification')
                elif cnt == len(dim_layers)-2 and output_type is None:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=None)
                elif cnt == len(dim_layers)-2 and output_type is 'same':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn)
                else:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn)
                layers.append(layer_tmp)
                params = params + para_tmp
    else:
        with tf.name_scope(tensorboard_name_scope):
            layers = []
            for cnt in range(len(dim_layers)-1):
                if cnt == 0:
                    layer_tmp, _ = new_fc_layer(net_input, dim_layers[0], dim_layers[1], activation_fn=activation_fn, weight=params[0], bias=params[1])
                elif cnt == len(dim_layers)-2 and output_type is 'classification':
                    layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn='classification', weight=params[2*cnt], bias=params[2*cnt+1])
                elif cnt == len(dim_layers)-2 and output_type is None:
                    layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1])
                elif cnt == len(dim_layers)-2 and output_type is 'same':
                    layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1])
                else:
                    layer_tmp, _ = new_fc_layer(layers[cnt-1], dim_layers[cnt], dim_layers[cnt+1], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1])
                layers.append(layer_tmp)
    return (layers, params)

############################################################
#####    functions for adding convolutional network    #####
############################################################
#### function to compute size of output from cnn_net
def cnn_net_output_size(input_size, k_sizes, stride_sizes, last_ch_size, padding_type='SAME', max_pool=False, pool_sizes=None, flat_output=False):
    num_layers = len(k_sizes)//2
    if not max_pool:
        pool_sizes = [1 for _ in range(len(k_sizes))]

    output_size = [input_size[0], input_size[1]]
    if padding_type is 'SAME':
        for cnt in range(num_layers):
            output_size[0] = ceil(float(output_size[0])/float(stride_sizes[2*cnt]))
            output_size[1] = ceil(float(output_size[1])/float(stride_sizes[2*cnt+1]))

            output_size[0] = int(ceil(float(output_size[0])/float(pool_sizes[2*cnt])))
            output_size[1] = int(ceil(float(output_size[1])/float(pool_sizes[2*cnt+1])))
    elif padding_type is 'VALID':
        for cnt in range(num_layers):
            output_size[0] = ceil(float(output_size[0]-k_sizes[2*cnt]+1)/float(stride_sizes[2*cnt]))
            output_size[1] = ceil(float(output_size[1]-k_sizes[2*cnt+1]+1)/float(stride_sizes[2*cnt+1]))

            output_size[0] = int(ceil(float(output_size[0]-pool_sizes[2*cnt]+1)/float(pool_sizes[2*cnt])))
            output_size[1] = int(ceil(float(output_size[1]-pool_sizes[2*cnt+1]+1)/float(pool_sizes[2*cnt+1])))
    else:
        output_size = [0, 0]
    if flat_output:
        return [output_size[0]*output_size[1]*last_ch_size]
    else:
        return output_size+[last_ch_size]

#### function to add 2D convolutional layer
def new_cnn_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None):
    with tf.name_scope('conv_layer'):
        if weight is None:
            weight = new_weight(shape=k_size)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]])

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type)
        if not (activation_fn is None):
            conv_layer = activation_fn(conv_layer + bias)

        if max_pooling:
            layer = tf.nn.max_pool(conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = conv_layer
    return (layer, [weight, bias])

#### function to generate network of convolutional layers
####      conv-pool-conv-pool-...-conv-pool-flat-dropout
####      k_sizes/stride_size/pool_sizes : [x_0, y_0, x_1, y_1, ..., x_m, y_m]
####      ch_sizes : [img_ch, ch_0, ch_1, ..., ch_m]
def new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0]):
    with tf.name_scope('conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_input, [])
        elif params is None:
            #### network & parameters are new
            layers, params = [], []
            for layer_cnt in range(len(k_sizes)//2):
                if layer_cnt == 0:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
                else:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
                layers.append(layer_tmp)
                params = params + para_tmp
        else:
            #### network generated from existing parameters
            layers = []
            for layer_cnt in range(len(k_sizes)//2):
                if layer_cnt == 0:
                    layer_tmp, _ = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
                else:
                    layer_tmp, _ = new_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
                layers.append(layer_tmp)

        #### flattening output
        output_dim = cnn_net_output_size(input_size, k_sizes, stride_sizes, ch_sizes[-1], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, flat_output=flat_output)
        if flat_output:
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, params, output_dim)

#### function to generate network of cnn->ffnn
def new_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None):
    cnn_model, cnn_params, cnn_output_dim = new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size)

    fc_model, fc_params = new_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type)
    return (cnn_model+fc_model, cnn_params, fc_params)