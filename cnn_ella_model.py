import numpy as np
import tensorflow as tf

from utils import *
#from utils_nn import *
from utils_ella_nn import *


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



#####################################################################
#####################################################################
#####################################################################


#### if linear relation, set 'relation_activation_fn' None
#### if nonlinear relation, set 'relation_activation_fn' with activation function such as tf.nn.relu
class ELLA_CNN_relation2_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, dim_img, dim_kernels, dim_strides, dim_cnn_know_base, dim_fc_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn=None):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 4*len(dim_channels)
        self.num_fc_KB_para = len(dim_fcs)
        self.num_fc_TS_para = 4*len(dim_fcs)
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.fc_KB_size = dim_fc_know_base
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, self.cnn_TS_param, self.fc_KB_param, self.fc_TS_param = new_ELLA_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=None, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn, fc_KB_params=None, fc_TS_params=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            else:
                model_tmp, _, cnn_TS_param_tmp, _, fc_TS_param_tmp = new_ELLA_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
                self.cnn_TS_param = self.cnn_TS_param + cnn_TS_param_tmp
                self.fc_TS_param = self.fc_TS_param + fc_TS_param_tmp
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + self.cnn_TS_param + self.fc_KB_param + self.fc_TS_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class ELLA_CNN_deconv_relation_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, dim_img, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, dim_fc_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, relation_activation_fn_fc=tf.nn.relu):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 3*len(dim_channels)
        self.num_fc_KB_para = len(dim_fcs)
        self.num_fc_TS_para = 4*len(dim_fcs)
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_KB_size = dim_fc_know_base
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, self.cnn_TS_param, self.fc_KB_param, self.fc_TS_param = new_ELLA_cnn_deconv_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn_fc, fc_KB_params=None, fc_TS_params=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            else:
                model_tmp, _, cnn_TS_param_tmp, _, fc_TS_param_tmp = new_ELLA_cnn_deconv_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn_fc, fc_KB_params=self.fc_KB_param, fc_TS_params=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
                self.cnn_TS_param = self.cnn_TS_param + cnn_TS_param_tmp
                self.fc_TS_param = self.fc_TS_param + fc_TS_param_tmp
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + self.cnn_TS_param + self.fc_KB_param + self.fc_TS_param

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (len(self.cnn_TS_param) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"
        assert (len(self.fc_KB_param) == self.num_fc_KB_para), "FFNN KB size doesn't match"
        assert (len(self.fc_TS_param) == self.num_tasks * self.num_fc_TS_para), "FFNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn_fc, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn_fc, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification', task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        self.cnn_KB_gradient = [tf.gradients(self.train_loss_and_reg[0], para) for para in self.cnn_KB_param[0:self.num_cnn_KB_para]]
        self.fc_KB_gradient = [tf.gradients(self.train_loss_and_reg[0], para) for para in self.fc_KB_param[0:self.num_fc_KB_para]]
        self.cnn_TS_gradient = [tf.gradients(self.train_loss_and_reg[0], para) for para in self.cnn_TS_param[0:self.num_cnn_TS_para]]
        self.fc_TS_gradient = [tf.gradients(self.train_loss_and_reg[0], para) for para in self.fc_TS_param[0:self.num_fc_TS_para]]