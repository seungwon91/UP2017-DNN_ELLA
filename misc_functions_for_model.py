import numpy as np
import tensorflow as tf


#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

#### function to generate weight parameter
def new_placeholder(shape):
    return tf.placeholder(shape=shape, dtype=tf.float32)

#### function to generate bias parameter
def new_weight_or_bias(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.2))

#### function to add fully-connected layer
def new_fc_layer(layer_input, input_dim, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None):
    if weight is None:
        weight = new_weight_or_bias(shape=[input_dim, output_dim])
    if bias is None:
        bias = new_weight_or_bias(shape=[output_dim])

    if activation_fn is None:
        layer = tf.matmul(layer_input, weight) + bias
    elif activation_fn is 'classification' and output_dim < 2:
        layer = tf.sigmoid(tf.matmul(layer_input, weight) + bias)
    elif activation_fn is 'classification':
        #layer = tf.nn.softmax(tf.matmul(layer_input, weight) + bias)
        layer = tf.matmul(layer_input, weight) + bias
    else:
        layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]



#### function to generate network of fully-connected layers
def new_fc_net(net_input, num_hiddens, activation_fn=tf.nn.relu, params=None, output_type=None):
    if params is None:
        layers, params = [], []
        for cnt in range(len(num_hiddens)-1):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(net_input, num_hiddens[0], num_hiddens[1], activation_fn=activation_fn)
            elif cnt == len(num_hiddens)-2 and output_type is 'classification':
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn='classification')
            elif cnt == len(num_hiddens)-2:
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn=None)
            else:
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn=activation_fn)
            layers.append(layer_tmp)
            params = params + para_tmp
    else:
        layers = []
        for cnt in range(len(num_hiddens)-1):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(net_input, num_hiddens[0], num_hiddens[1], activation_fn=activation_fn, weight=params[0], bias=params[1])
            elif cnt == len(num_hiddens)-2 and output_type is 'classification':
                layer_tmp, _ = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn='classification', weight=params[2*cnt], bias=params[2*cnt+1])
            elif cnt == len(num_hiddens)-2:
                layer_tmp, _ = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1])
            else:
                layer_tmp, _ = new_fc_layer(layers[cnt-1], num_hiddens[cnt], num_hiddens[cnt+1], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1])
            layers.append(layer_tmp)
    return (layers, params)



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

