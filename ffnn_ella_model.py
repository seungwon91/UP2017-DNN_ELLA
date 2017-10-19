import numpy as np
import tensorflow as tf

from utils import *
from utils_nn import *

#################################################
############ Miscellaneous Functions ############
#################################################
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

#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

############################################
###### functions for ELLA-FFNN simple ######
############################################
#### function to generate knowledge-base parameters for ELLA_simple layer
def new_KB_simple_param(dim_kb, input_dim, output_dim, layer_number, reg_type):
    w_name, b_name = 'KB_W'+str(layer_number), 'KB_b'+str(layer_number)
    return [tf.get_variable(name=w_name, shape=[dim_kb[0], input_dim, output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=b_name, shape=[dim_kb[1], output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to generate task-specific parameters for ELLA_simple layer
def new_TS_simple_param(dim_kb, layer_number, task_number, reg_type):
    sw_name, sb_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=sw_name, shape=[1, dim_kb[0]], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=sb_name, shape=[1, dim_kb[1]], dtype=tf.float32, regularizer=reg_type)]

#### function to add ELLA_simple layer
def new_ELLA_simple_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number,  activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_simple_param(KB_dim, input_dim, output_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_simple_param(KB_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        W, b = tf.tensordot(TS_param[2*task_cnt], KB_param[0], axes=1)[0], tf.tensordot(TS_param[2*task_cnt+1], KB_param[1], axes=1)[0]
        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        elif activation_fn is 'classification' and output_dim < 2:
            layer_eqn.append( tf.sigmoid(tf.matmul(layer_input_list[task_cnt], W)+b) )
        elif activation_fn is 'classification':
            #layer_eqn.append( tf.nn.softmax(tf.matmul(layer_input_list[task_cnt], W)+b) )
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, KB_param, TS_param

############################################
####      functions for ELLA-FFNN       ####
####  nonlinear relation btw KB and TS  ####
############################################
#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_KB_nonlinear_relation_param(dim_kb, layer_number, reg_type):
    kb_name = 'KB_'+str(layer_number)
    return tf.get_variable(name=kb_name, shape=[1, dim_kb], dtype=tf.float32, regularizer=reg_type)

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_TS_nonlinear_relation_param(dim_kb, input_dim, output_dim, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=[dim_kb, (input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=[(input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to add ELLA_tensorfactor layer
#def new_ELLA_nonlinear_relation_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.tanh, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
def new_ELLA_nonlinear_relation_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_nonlinear_relation_param(KB_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_nonlinear_relation_param(KB_dim, input_dim, output_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        if para_activation_fn is None:
            para_tmp = tf.matmul(KB_param, TS_param[2*task_cnt]) + TS_param[2*task_cnt+1]
        else:
            para_tmp = para_activation_fn(tf.matmul(KB_param, TS_param[2*task_cnt]) + TS_param[2*task_cnt+1])

        W_tmp, b = tf.split(tf.reshape(para_tmp, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, [KB_param], TS_param


#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_TS_nonlinear_relation_param2(dim_kb, dim_ts, input_dim, output_dim, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number), 'TS_K'+str(layer_number)+'_'+str(task_number), 'TS_p'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=[dim_kb, dim_ts], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=[dim_ts], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=[dim_ts, (input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=[(input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to add ELLA_tensorfactor layer
def new_ELLA_nonlinear_relation_layer2(layer_input_list, input_dim, output_dim, KB_dim, TS_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_nonlinear_relation_param(KB_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_nonlinear_relation_param2(KB_dim, TS_dim, input_dim, output_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        if para_activation_fn is None:
            para_tmp = tf.matmul(tf.matmul(KB_param, TS_param[4*task_cnt]) + TS_param[4*task_cnt+1], TS_param[4*task_cnt+2]) + TS_param[4*task_cnt+3]
        else:
            para_tmp = tf.matmul(para_activation_fn(tf.matmul(KB_param, TS_param[4*task_cnt]) + TS_param[4*task_cnt+1]), TS_param[4*task_cnt+2]) + TS_param[4*task_cnt+3]

        W_tmp, b = tf.split(tf.reshape(para_tmp, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, [KB_param], TS_param

#########################################################
####   ELLA Neural Network for Multi-task Learning   ####
####      stacked in higher dim tensor, and sum      ####
#########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB_W0, KB_b0, KB_W1, KB_b1, ...]
class ELLA_FFNN_simple_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        if self.num_tasks < 2:
            self.train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            self.train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            self.valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            self.valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            self.test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            self.test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            self.train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(self.num_tasks)]

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_simple_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_simple_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_simple_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_simple_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_simple_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_simple_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_simple_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_simple_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_simple_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        self.train_loss_list = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
        self.train_loss_and_reg_list = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) + reg_term1 + reg_term2 for x in range(self.num_tasks)]
        self.valid_loss_list = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
        self.test_loss_list = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg_list[x]) for x in range(self.num_tasks)]


########################################################
####   ELLA Neural Network for Multi-task Learning  ####
####         linear relation btw KB and TS          ####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB0, KB1, ...]
class ELLA_FFNN_linear_relation_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        if self.num_tasks < 2:

            self.train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            self.train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            self.valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            self.valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            self.test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            self.test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            self.train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(self.num_tasks)]

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data

        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]


class ELLA_FFNN_linear_relation_minibatch2():
    def __init__(self, num_tasks, dim_layers, dim_know_base, dim_task_specific, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.TS_size = dim_task_specific
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        if self.num_tasks < 2:

            self.train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            self.train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            self.valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            self.valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            self.test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            self.test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            self.train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(self.num_tasks)]

            self.train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(self.num_tasks)]

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]





########################################################
####   ELLA Neural Network for Multi-task Learning  ####
####       nonlinear relation btw KB and TS         ####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB0, KB1, ...]
class ELLA_FFNN_nonlinear_relation_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        if self.num_tasks < 2:

            self.train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            self.train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            self.valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            self.valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            self.test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            self.test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            self.train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(self.num_tasks)]

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]







class ELLA_FFNN_nonlinear_relation_minibatch2():
    def __init__(self, num_tasks, dim_layers, dim_know_base, dim_task_specific, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.TS_size = dim_task_specific
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        if self.num_tasks < 2:

            self.train_x = [tf.constant(data_list[0], dtype=tf.float32)]
            self.train_y = [tf.constant(data_list[1], dtype=tf.float32)]
            self.valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
            self.valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
            self.test_x = [tf.constant(data_list[4], dtype=tf.float32)]
            self.test_y = [tf.constant(data_list[5], dtype=tf.float32)]
        else:
            self.train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(self.num_tasks)]

            self.train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(self.num_tasks)]
            self.test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(self.num_tasks)]

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]


