import numpy as np
import tensorflow as tf


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
        if self.num_tasks < 2:
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



############################################
#####   functions for adding network   #####
############################################
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
