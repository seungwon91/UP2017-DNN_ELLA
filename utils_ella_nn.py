import numpy as np
import tensorflow as tf

from utils import *
from utils_nn import *


############################################################
#####   functions for adding ELLA network (CNN ver)    #####
############################################################
#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_ELLA_KB_param(shape, layer_number, reg_type):
    kb_name = 'KB_'+str(layer_number)
    return tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type)

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_TS_param(shape, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_W0_'+str(layer_number)+'_'+str(task_number), 'TS_b0_'+str(layer_number)+'_'+str(task_number), 'TS_W1_'+str(layer_number)+'_'+str(task_number), 'TS_b1_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[3], dtype=tf.float32, regularizer=reg_type)]

#### function to generate convolutional layer with shared knowledge base
def new_ELLA_cnn_layer(layer_input, k_size, ch_size, stride_size, KB_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None):
    with tf.name_scope('ELLA_conv_shared_KB'):
        if KB_param is None:
            ## KB \in R^{h \times c}
            KB_param = new_ELLA_KB_param(KB_size, layer_num, KB_reg_type)
        if TS_param is None:
            ## TS1 \in R^{(H*W*Ch_in+1) \times h}
            ## TS2 \in R^{c \times Ch_out}
            ## tensordot(KB, TS1) -> R^{c \times (H*W*Ch_in+1)}
            ## tensordot(..., TS2) -> R^{(H*W*Ch_in+1) \times Ch_out}
            TS_param = new_ELLA_TS_param([[k_size[0]*k_size[1]*ch_size[0]+1, KB_size[0]], [1, k_size[0]*k_size[1]*ch_size[0]+1], [KB_size[1], ch_size[1]], [1, ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_conv_TS'):
        if para_activation_fn is None:
            para_tmp = tf.add(tf.tensordot(KB_param, TS_param[0], [[0], [1]]), TS_param[1])
        else:
            para_tmp = para_activation_fn(tf.add(tf.tensordot(KB_param, TS_param[0], [[0], [1]]), TS_param[1]))
        para_last = tf.add(tf.tensordot(para_tmp, TS_param[2], [[0], [0]]), TS_param[3])

        W_tmp, b = tf.split(tf.reshape(para_last, [(k_size[0]*k_size[1]*ch_size[0]+1)*ch_size[1]]), [k_size[0]*k_size[1]*ch_size[0]*ch_size[1], ch_size[1]])
        W = tf.reshape(W_tmp, [k_size[0], k_size[1], ch_size[0], ch_size[1]])

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0):
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    with tf.name_scope('ELLA_conv_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])

            layers.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

        #### flattening output
        output_dim = cnn_net_output_size(input_size, k_sizes, stride_sizes, ch_sizes[-1], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, flat_output=flat_output)
        if flat_output:
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, output_dim)


############################################################
#####   functions for adding ELLA network (FFNN ver)   #####
############################################################
#### function to generate fully connected layer with shared knowledge base
def new_ELLA_fc_layer(layer_input, input_dim, output_dim, KB_dim, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    with tf.name_scope('ELLA_fc_shared_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h}
            KB_param = new_ELLA_KB_param([1, KB_dim[0]], layer_num, KB_reg_type)
        if TS_param is None:
            ## TS1 \in R^{h \times c}
            ## TS2 \in R^{c \times ((Cin+1)*Cout)}
            TS_param = new_ELLA_TS_param([KB_dim, [1, KB_dim[1]], [KB_dim[1], (input_dim+1)*output_dim], [1, (input_dim+1)*output_dim]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_fc_TS'):
        if para_activation_fn is None:
            para_tmp = tf.add(tf.matmul(KB_param, TS_param[0]), TS_param[1])
        else:
            para_tmp = para_activation_fn(tf.add(tf.matmul(KB_param, TS_param[0]), TS_param[1]))
        para_last = tf.add(tf.matmul(para_tmp, TS_param[2]), TS_param[3])

        W_tmp, b = tf.split(tf.reshape(para_last, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

    layer_eqn, _ = new_fc_layer(layer_input, input_dim, output_dim, activation_fn=activation_fn, weight=W, bias=b)
    return layer_eqn, [KB_param], TS_param


#### function to generate network of fully connected layers with shared knowledge base
def new_ELLA_fc_net(net_input, dim_layers, dim_KBs, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, output_type=None, layer_start_index=0, task_index=0):
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    with tf.name_scope('ELLA_fc_net'):
        layers = []
        for layer_cnt in range(len(dim_layers)-1):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[4*layer_cnt:4*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)

            layers.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

    return (layers, KB_params, TS_params)


#### function to generate network of cnn->ffnn
def new_ELLA_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, fc_KB_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_para_activation_fn=tf.nn.relu, fc_KB_params=None, fc_TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_output_dim = new_ELLA_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index)

    ## add fc layers
    fc_model, fc_KB_params, fc_TS_params = new_ELLA_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, fc_KB_sizes, activation_fn=fc_activation_fn, para_activation_fn=fc_para_activation_fn, KB_params=fc_KB_params, TS_params=fc_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, output_type=output_type, layer_start_index=len(k_sizes)//2, task_index=task_index)

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, fc_KB_params, fc_TS_params)



###############################################################
##### functions for adding ELLA network (CNN/Deconv ver)  #####
###############################################################
#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_cnn_deconv_TS_param(shape, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type)]

#### function to generate convolutional layer with shared knowledge base
#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : deconv_filter_height(and width)
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_ELLA_cnn_deconv_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h \times w \times c}
            KB_param = new_ELLA_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, KB_reg_type)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times ch_in*ch_out \times c}
            ## TS2 : Deconv bias \in R^{ch_out}
            TS_param = new_ELLA_cnn_deconv_TS_param([[TS_size, TS_size, ch_size[0]*ch_size[1], KB_size[1]], [1, 1, 1, ch_size[0]*ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], ch_size[0]*ch_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)

        W, b = tf.reshape(para_tmp, k_size+ch_size), TS_param[2]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_deconv_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0):
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[3*layer_cnt:3*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[3*layer_cnt:3*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])

            layers.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

        #### flattening output
        output_dim = cnn_net_output_size(input_size, k_sizes, stride_sizes, ch_sizes[-1], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, flat_output=flat_output)

        if flat_output:
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, output_dim)


#### function to generate network of cnn->ffnn
def new_ELLA_cnn_deconv_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, fc_KB_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_para_activation_fn=tf.nn.relu, fc_KB_params=None, fc_TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_output_dim = new_ELLA_cnn_deconv_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index)

    ## add fc layers
    fc_model, fc_KB_params, fc_TS_params = new_ELLA_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, fc_KB_sizes, activation_fn=fc_activation_fn, para_activation_fn=fc_para_activation_fn, KB_params=fc_KB_params, TS_params=fc_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, output_type=output_type, layer_start_index=len(k_sizes)//2, task_index=task_index)

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, fc_KB_params, fc_TS_params)

























############################################
##### functions for tensor factor FFNN #####
############################################
#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_KB_tensorfactor_param(dim_kb, input_dim, output_dim, layer_number, reg_type):
    w_name, b_name = 'KB_W'+str(layer_number), 'KB_b'+str(layer_number)
    return [tf.get_variable(name=w_name, shape=[input_dim, dim_kb[0]], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=b_name, shape=[output_dim, dim_kb[1]], dtype=tf.float32, regularizer=reg_type)]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_TS_tensorfactor_param(dim_kb, output_dim, layer_number, task_number, reg_type):
    sw_name, sb_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=sw_name, shape=[dim_kb[0], output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=sb_name, shape=[dim_kb[1], 1], dtype=tf.float32, regularizer=reg_type)]

#### function to add ELLA_tensorfactor layer
def new_ELLA_tensorfactor_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number,  activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_tensorfactor_param(KB_dim, input_dim, output_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_tensorfactor_param(KB_dim, output_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        W, b = tf.matmul(KB_param[0], TS_param[2*task_cnt]), tf.matmul(KB_param[1], TS_param[2*task_cnt+1])[0]
        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, KB_param, TS_param
