import numpy as np
import tensorflow as tf

from utils import *
from utils_nn import *

#################################################
#########       Simple CNN batch       ##########
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_batch():
    def __init__(self, dim_channels, dim_fcs, dim_img, dim_kernel, dim_strides, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*len(channels_size-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*len(channels_size-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay

        #### placeholder of model
        self.model_input = new_placeholder([None, self.image_size[0]*self.image_size[1]*self.image_size[2]])
        self.true_output = new_placeholder([None, self.fc_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        reshaped_input = tf.reshape(self.model_input, [-1]+self.image_size)

        #### layers of model
        self.layers, self.cnn_param, self.fc_param = new_cnn_fc_net(reshaped_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
        self.param = self.cnn_param + self.fc_param

        #### functions of model
        if self.fc_size[-1]>1:
            self.eval = tf.nn.softmax(self.layers[-1])
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.true_output, logits=self.layers[-1])
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.eval, 1), tf.argmax(self.true_output, 1)), tf.float32))
        else:
            self.eval = self.layers[-1]
            self.loss = 2.0* tf.nn.l2_loss(self.eval-self.true_output)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal((self.eval>0.5), (self.true_output>0.5)), tf.float32))

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.loss)


#################################################
#######      Simple CNN mini-batch       ########
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_minibatch():
    def __init__(self, dim_channels, dim_fcs, dim_img, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=False)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=False)

        #### layers of model for train data
        self.train_layers, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_layers, _, _ = new_cnn_fc_net(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')

        #### layers of model for test data
        self.test_layers, _, _ = new_cnn_fc_net(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')

        #### functions of model
        tr_eval, v_eval, test_eval, tr_loss, v_loss, test_loss, tr_acc, v_acc, test_acc = mtl_model_output_functions(models=[[self.train_layers], [self.valid_layers], [self.test_layers]], y_batches=[[train_y_batch], [valid_y_batch], [test_y_batch]], num_tasks=1, dim_output=self.fc_size[-1], classification=True)

        self.train_eval, self.valid_eval, self.test_eval = tr_eval[0], v_eval[0], test_eval[0]
        self.train_loss, self.valid_loss, self.test_loss = tr_loss[0], v_loss[0], test_loss[0]
        self.train_accuracy, self.valid_accuracy, self.test_accuracy = tr_acc[0], v_acc[0], test_acc[0]

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)


########################################################
####     Single CNN + FC for Multi-task Learning    ####
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, dim_img, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            else:
                model_tmp, _, _ = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.train_models.append(model_tmp)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]


'''
########################################################
####     Single CNN + FC for Multi-task Learning    ####
########         Hard Parameter Sharing         ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, dim_img, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        self.num_tasks = num_tasks
        #self.num_layers = [len(dim_fcs[0])] + [len(dim_fcs[1][x]) for x in range(self.num_tasks)]
        self.cnn_channels_size = [dim_img[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        #self.fc_size = dim_fcs
        self.shared_fc_size = dim_fcs[0]
        self.task_specific_fc_size = dim_fcs[1]
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            else:
                model_tmp, _, _ = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.train_models.append(model_tmp)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.image_size[0:2], output_type='classification')
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]
'''
