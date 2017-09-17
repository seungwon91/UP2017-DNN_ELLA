import numpy as np
import tensorflow as tf
from misc_functions_for_model import *

#################################################
############       Simple CNN       #############
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_batch():
    def __init__(self, dim_channels, dim_fcs, dim_img, dim_kernel, learning_rate, learning_rate_decay, max_pooling=False, dim_pool=None):
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)-1
        self.channels_size = dim_channels    ## include dim of input channel
        self.kernel_size = dim_kernel     ## len(kernel_size) == 2*len(channels_size-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.pool_size = dim_pool      ## len(pool_size) == 2*len(channels_size-1)

        #### placeholder of model
        self.model_input = new_placeholder([None, self.image_size[0]*self.image_size[1]*self.image_size[2]])
        self.true_output = new_placeholder([None, self.fc_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)

        reshaped_input = tf.reshape(self.model_input, [-1]+self.image_size)

        #### layers of model
        self.layers, self.param = [], []
        cnn_img_size = self.image_size[0:2]
        for cnt in range(len(self.channels_size)-1):
            if cnt == 0:
                layer_tmp, para_tmp = new_cnn_layer(layer_input=reshaped_input, k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1])
            else:
                layer_tmp, para_tmp = new_cnn_layer(layer_input=self.layers[cnt-1], k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1])
            self.layers.append(layer_tmp)
            self.param = self.param + para_tmp
            if max_pooling:
                cnn_img_size = [x//y for x, y in zip(cnn_img_size, self.pool_size[2*cnt:2*(cnt+1)])]

        input_dim = cnn_img_size[0]*cnn_img_size[1]*self.channels_size[-1]
        self.layers.append(tf.reshape(self.layers[-1], [-1, input_dim]))
        for cnt in range(len(self.fc_size)):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(self.layers[len(self.channels_size)+cnt-1], input_dim, self.fc_size[cnt])
            if cnt == len(self.fc_size)-1:
                layer_tmp, para_tmp = new_fc_layer(self.layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], activation_fn='classification')
            else:
                layer_tmp, para_tmp = new_fc_layer(self.layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt])
            self.layers.append(layer_tmp)
            self.param = self.param + para_tmp

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
############       Simple CNN       #############
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_minibatch():
    def __init__(self, dim_channels, dim_fcs, dim_img, dim_kernel, batch_size, learning_rate, learning_rate_decay, data_list, max_pooling=False, dim_pool=None):
        #### num_layers == len(channels_size) + len(fc_size) -1 && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)-1
        self.channels_size = dim_channels    ## include dim of input channel
        self.kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=False)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=False)

        #### layers of model for train data
        self.train_layers, self.param, cnn_img_size = [], [], self.image_size[0:2]
        for cnt in range(len(self.channels_size)-1):
            if cnt == 0:
                layer_tmp, para_tmp = new_cnn_layer(layer_input=train_x_batch, k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1])
            else:
                layer_tmp, para_tmp = new_cnn_layer(layer_input=self.train_layers[cnt-1], k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1])
            self.train_layers.append(layer_tmp)
            self.param = self.param + para_tmp
            if max_pooling:
                cnn_img_size = [x//y for x, y in zip(cnn_img_size, self.pool_size[2*cnt:2*(cnt+1)])]

        input_dim = cnn_img_size[0]*cnn_img_size[1]*self.channels_size[-1]
        self.train_layers.append(tf.reshape(self.train_layers[-1], [-1, input_dim]))
        for cnt in range(len(self.fc_size)):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(self.train_layers[len(self.channels_size)+cnt-1], input_dim, self.fc_size[cnt])
            elif cnt == len(self.fc_size)-1:
                layer_tmp, para_tmp = new_fc_layer(self.train_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], activation_fn='classification')
            else:
                layer_tmp, para_tmp = new_fc_layer(self.train_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt])
            self.train_layers.append(layer_tmp)
            self.param = self.param + para_tmp

        #### layers of model for validation data
        self.valid_layers = []
        for cnt in range(len(self.channels_size)-1):
            if cnt == 0:
                layer_tmp, _ = new_cnn_layer(layer_input=valid_x_batch, k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1], weight=self.param[0], bias=self.param[1])
            else:
                layer_tmp, _ = new_cnn_layer(layer_input=self.valid_layers[cnt-1], k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1], weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            self.valid_layers.append(layer_tmp)

        self.valid_layers.append(tf.reshape(self.valid_layers[-1], [-1, input_dim]))
        for cnt in range(len(self.fc_size)):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(self.valid_layers[len(self.channels_size)+cnt-1], input_dim, self.fc_size[cnt], weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            elif cnt == len(self.fc_size)-1:
                layer_tmp, _ = new_fc_layer(self.valid_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], activation_fn='classification', weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            else:
                layer_tmp, _ = new_fc_layer(self.valid_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            self.valid_layers.append(layer_tmp)

        #### layers of model for test data
        self.test_layers = []
        for cnt in range(len(self.channels_size)-1):
            if cnt == 0:
                layer_tmp, _ = new_cnn_layer(layer_input=test_x_batch, k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1], weight=self.param[0], bias=self.param[1])
            else:
                layer_tmp, _ = new_cnn_layer(layer_input=self.test_layers[cnt-1], k_size=self.kernel_size[2*cnt:2*(cnt+1)]+self.channels_size[cnt:cnt+2], max_pooling=max_pooling, pool_size=[1]+self.pool_size[2*cnt:2*(cnt+1)]+[1], weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            self.test_layers.append(layer_tmp)

        self.test_layers.append(tf.reshape(self.test_layers[-1], [-1, input_dim]))
        for cnt in range(len(self.fc_size)):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(self.test_layers[len(self.channels_size)+cnt-1], input_dim, self.fc_size[cnt], weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            elif cnt == len(self.fc_size)-1:
                layer_tmp, _ = new_fc_layer(self.test_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], activation_fn='classification', weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            else:
                layer_tmp, _ = new_fc_layer(self.test_layers[len(self.channels_size)+cnt-1], self.fc_size[cnt-1], self.fc_size[cnt], weight=self.param[2*(len(self.channels_size)+cnt-1)], bias=self.param[2*(len(self.channels_size)+cnt-1)+1])
            self.test_layers.append(layer_tmp)

        #### functions of model
        tr_eval, v_eval, test_eval, tr_loss, v_loss, test_loss, tr_acc, v_acc, test_acc = mtl_model_output_functions(models=[[self.train_layers], [self.valid_layers], [self.test_layers]], y_batches=[[train_y_batch], [valid_y_batch], [test_y_batch]], num_tasks=1, dim_output=self.fc_size[-1], classification=True)

        self.train_eval, self.valid_eval, self.test_eval = tr_eval[0], v_eval[0], test_eval[0]
        self.train_loss, self.valid_loss, self.test_loss = tr_loss[0], v_loss[0], test_loss[0]
        self.train_accuracy, self.valid_accuracy, self.test_accuracy = tr_acc[0], v_acc[0], test_acc[0]

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)
        #self.update = tf.train.AdamOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)
