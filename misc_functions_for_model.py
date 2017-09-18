import numpy as np
import tensorflow as tf
from math import ceil

############################################
#####     functions for saving data    #####
############################################
def tflized_data(data_list, do_MTL, num_tasks=0):
    if not do_MTL:
        #### single task
        assert (len(data_list) == 6), "Data given to model isn't in the right format"
        train_x = tf.constant(data_list[0], dtype=tf.float32)
        train_y = tf.constant(data_list[1], dtype=tf.float32)
        valid_x = tf.constant(data_list[2], dtype=tf.float32)
        valid_y = tf.constant(data_list[3], dtype=tf.float32)
        test_x = tf.constant(data_list[4], dtype=tf.float32)
        test_y = tf.constant(data_list[5], dtype=tf.float32)
    else:
        #### multi-task
        if num_tasks < 2:
            train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(num_tasks)]
            train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(num_tasks)]
            valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(num_tasks)]
            valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(num_tasks)]
            test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(num_tasks)]
            test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(num_tasks)]
    return [train_x, train_y, valid_x, valid_y, test_x, test_y]


def minibatched_data(data_list, batch_size, data_index, do_MTL, num_tasks=0):
    if not do_MTL:
        #### single task
        train_x_batch = tf.slice(data_list[0], [batch_size * data_index, 0], [batch_size, -1])
        train_y_batch = tf.slice(data_list[1], [batch_size * data_index, 0], [batch_size, -1])
        valid_x_batch = tf.slice(data_list[2], [batch_size * data_index, 0], [batch_size, -1])
        valid_y_batch = tf.slice(data_list[3], [batch_size * data_index, 0], [batch_size, -1])
        test_x_batch = tf.slice(data_list[4], [batch_size * data_index, 0], [batch_size, -1])
        test_y_batch = tf.slice(data_list[5], [batch_size * data_index, 0], [batch_size, -1])
    else:
        #### multi-task
        train_x_batch = [tf.slice(data_list[0][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        train_y_batch = [tf.slice(data_list[1][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        valid_x_batch = [tf.slice(data_list[2][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        valid_y_batch = [tf.slice(data_list[3][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        test_x_batch = [tf.slice(data_list[4][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        test_y_batch = [tf.slice(data_list[5][x], [batch_size * data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)

def minibatched_cnn_data(data_list, batch_size, data_index, data_tensor_dim, do_MTL, num_tasks=0):
    if not do_MTL:
        #### single task
        train_x_batch = tf.reshape(tf.slice(data_list[0], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
        train_y_batch = tf.slice(data_list[1], [batch_size*data_index, 0], [batch_size, -1])
        valid_x_batch = tf.reshape(tf.slice(data_list[2], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
        valid_y_batch = tf.slice(data_list[3], [batch_size*data_index, 0], [batch_size, -1])
        test_x_batch = tf.reshape(tf.slice(data_list[4], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
        test_y_batch = tf.slice(data_list[5], [batch_size*data_index, 0], [batch_size, -1])
    else:
        #### multi-task
        train_x_batch = [tf.reshape(tf.slice(data_list[0][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
        train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        valid_x_batch = [tf.reshape(tf.slice(data_list[2][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
        valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
        test_x_batch = [tf.reshape(tf.slice(data_list[4][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
        test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


############################################
#### functions for (MTL) model's output ####
############################################
def mtl_model_output_functions(models, y_batches, num_tasks, dim_output, classification=False):
    if classification and dim_output > 1:
        train_eval = [tf.nn.softmax(models[0][x][-1]) for x in range(num_tasks)]
        valid_eval = [tf.nn.softmax(models[1][x][-1]) for x in range(num_tasks)]
        test_eval = [tf.nn.softmax(models[2][x][-1]) for x in range(num_tasks)]

        train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=y_batches[0][x], logits=models[0][x][-1]) for x in range(num_tasks)]
        valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=y_batches[1][x], logits=models[1][x][-1]) for x in range(num_tasks)]
        test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=y_batches[2][x], logits=models[2][x][-1]) for x in range(num_tasks)]

        train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(train_eval[x], 1), tf.argmax(y_batches[0][x], 1)), tf.float32)) for x in range(num_tasks)]
        valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(valid_eval[x], 1), tf.argmax(y_batches[1][x], 1)), tf.float32)) for x in range(num_tasks)]
        test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test_eval[x], 1), tf.argmax(y_batches[2][x], 1)), tf.float32)) for x in range(num_tasks)]
    else:
        train_eval = [models[0][x][-1] for x in range(num_tasks)]
        valid_eval = [models[1][x][-1] for x in range(num_tasks)]
        test_eval = [models[2][x][-1] for x in range(num_tasks)]

        train_loss = [2.0* tf.nn.l2_loss(train_eval[x]-y_batches[0][x]) for x in range(num_tasks)]
        valid_loss = [2.0* tf.nn.l2_loss(valid_eval[x]-y_batches[1][x]) for x in range(num_tasks)]
        test_loss = [2.0* tf.nn.l2_loss(test_eval[x]-y_batches[2][x]) for x in range(num_tasks)]

        if classification:
            train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((train_eval[x]>0.5), (y_batches[0][x]>0.5)), tf.float32)) for x in range(num_tasks)]
            valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((valid_eval[x]>0.5), (y_batches[1][x]>0.5)), tf.float32)) for x in range(num_tasks)]
            test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((test_eval[x]>0.5), (y_batches[2][x]>0.5)), tf.float32)) for x in range(num_tasks)]
        else:
            train_accuracy, valid_accuracy, test_accuracy = None, None, None
    return (train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy)



############################################################
#####   functions for adding fully-connected network   #####
############################################################
#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

#### function to generate weight parameter
def new_placeholder(shape):
    return tf.placeholder(shape=shape, dtype=tf.float32)

#### function to generate weight parameter
def new_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.2))

#### function to generate bias parameter
def new_bias(shape):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=shape))

#### function to add fully-connected layer
def new_fc_layer(layer_input, input_dim, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None):
    if weight is None:
        weight = new_weight(shape=[input_dim, output_dim])
    if bias is None:
        bias = new_bias(shape=[output_dim])

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
####      'dim_layers' contains input/output layer
def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None):
    if len(dim_layers) < 2:
        #### for the case that hard-parameter shared network does not have shared layers
        return (net_input, [])
    elif params is None:
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
