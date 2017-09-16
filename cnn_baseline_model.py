import numpy as np
import tensorflow as tf

#################################################
############ Miscellaneous Functions ############
#################################################
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

#### function to add 2D convolutional layer
def new_cnn_layer(layer_input, k_size, activation_fn=tf.nn.relu, weight=None, bias=None, max_pooling=False, pool_size=None):
    if weight is None:
        weight = new_weight(shape=k_size)
    if bias is None:
        bias = new_bias(shape=[k_size[-1]])

    cnn_layer = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
    if not (activation_fn is None):
        cnn_layer = activation_fn(cnn_layer + bias)

    if max_pooling:
        layer = tf.nn.max_pool(cnn_layer, ksize=pool_size, strides=pool_size, padding='SAME')
    else:
        layer = cnn_layer
    return layer, [weight, bias]
        
#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

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
def new_tensorfactor_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number,  activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
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
        self.kernel_size = dim_kernel     ## len(kernel_size) == 2*len(channels_size-1)
        self.fc_size = dim_fcs
        self.image_size = dim_img    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.pool_size = dim_pool      ## len(pool_size) == 2*len(channels_size-1)
        self.batch_size = batch_size

        #### data
        self.train_x = tf.constant(data_list[0], dtype=tf.float32)
        self.train_y = tf.constant(data_list[1], dtype=tf.float32)
        self.valid_x = tf.constant(data_list[2], dtype=tf.float32)
        self.valid_y = tf.constant(data_list[3], dtype=tf.float32)
        self.test_x = tf.constant(data_list[4], dtype=tf.float32)
        self.test_y = tf.constant(data_list[5], dtype=tf.float32)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch = tf.reshape(tf.slice(self.train_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1]), [-1]+self.image_size)
        train_y_batch = tf.slice(self.train_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        valid_x_batch = tf.reshape(tf.slice(self.valid_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1]), [-1]+self.image_size)
        valid_y_batch = tf.slice(self.valid_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        test_x_batch = tf.reshape(tf.slice(self.test_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1]), [-1]+self.image_size)
        test_y_batch = tf.slice(self.test_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])

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
        if self.fc_size[-1]>1:
            self.train_eval = tf.nn.softmax(self.train_layers[-1])
            self.valid_eval = tf.nn.softmax(self.valid_layers[-1])
            self.test_eval = tf.nn.softmax(self.test_layers[-1])

            self.train_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch, logits=self.train_layers[-1])
            self.valid_loss = tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch, logits=self.valid_layers[-1])
            self.test_loss = tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch, logits=self.test_layers[-1])

            self.train_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval, 1), tf.argmax(train_y_batch, 1)), tf.float32))
            self.valid_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval, 1), tf.argmax(valid_y_batch, 1)), tf.float32))
            self.test_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval, 1), tf.argmax(test_y_batch, 1)), tf.float32))
        else:
            self.train_eval = self.train_layers[-1]
            self.valid_eval = self.valid_layers[-1]
            self.test_eval = self.test_layers[-1]

            self.train_loss = 2.0* tf.nn.l2_loss(self.train_eval-train_y_batch)
            self.valid_loss = 2.0* tf.nn.l2_loss(self.valid_eval-valid_y_batch)
            self.test_loss = 2.0* tf.nn.l2_loss(self.test_eval-test_y_batch)

            self.train_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.train_eval>0.5), (train_y_batch>0.5)), tf.float32))
            self.valid_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.valid_eval>0.5), (valid_y_batch>0.5)), tf.float32))
            self.test_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.test_eval>0.5), (test_y_batch>0.5)), tf.float32))

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)
        #self.update = tf.train.AdamOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)




































#### Feedforward Neural Network - mini batch ver.
class FFNN_minibatch():
    def __init__(self, dim_layers, batch_size, learning_rate, learning_rate_decay, data_list, classification=False):
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.train_x = tf.constant(data_list[0], dtype=tf.float32)
        self.train_y = tf.constant(data_list[1], dtype=tf.float32)
        self.valid_x = tf.constant(data_list[2], dtype=tf.float32)
        self.valid_y = tf.constant(data_list[3], dtype=tf.float32)
        self.test_x = tf.constant(data_list[4], dtype=tf.float32)
        self.test_y = tf.constant(data_list[5], dtype=tf.float32)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch = tf.slice(self.train_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        train_y_batch =tf.slice(self.train_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        valid_x_batch = tf.slice(self.valid_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        valid_y_batch =tf.slice(self.valid_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        test_x_batch = tf.slice(self.test_x, [self.batch_size * self.data_index, 0], [self.batch_size, -1])
        test_y_batch =tf.slice(self.test_y, [self.batch_size * self.data_index, 0], [self.batch_size, -1])

        #### layers of model for train data
        self.train_layers, self.param = [], []
        for cnt in range(self.num_layers):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(train_x_batch, self.layers_size[0], self.layers_size[1])
            elif cnt == self.num_layers-1:
                layer_tmp, para_tmp = new_fc_layer(self.train_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], activation_fn=None)
            else:
                layer_tmp, para_tmp = new_fc_layer(self.train_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1])
            self.train_layers.append(layer_tmp)
            self.param = self.param + para_tmp

        #### layers of model for validation data
        self.valid_layers = []
        for cnt in range(self.num_layers):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(valid_x_batch, self.layers_size[0], self.layers_size[1], weight=self.param[0], bias=self.param[1])
            elif cnt == self.num_layers-1:
                layer_tmp, _ = new_fc_layer(self.valid_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], activation_fn=None, weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            else:
                layer_tmp, _ = new_fc_layer(self.valid_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            self.valid_layers.append(layer_tmp)

        #### layers of model for test data
        self.test_layers = []
        for cnt in range(self.num_layers):
            if cnt == 0:
                layer_tmp, _ = new_fc_layer(test_x_batch, self.layers_size[0], self.layers_size[1], weight=self.param[0], bias=self.param[1])
            elif cnt == self.num_layers-1:
                layer_tmp, _ = new_fc_layer(self.test_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], activation_fn=None, weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            else:
                layer_tmp, _ = new_fc_layer(self.test_layers[cnt-1], self.layers_size[cnt], self.layers_size[cnt+1], weight=self.param[2*cnt], bias=self.param[2*cnt+1])
            self.test_layers.append(layer_tmp)

        #### functions of model
        if classification and self.layers_size[-1]>1:
            self.train_eval = tf.nn.softmax(self.train_layers[-1])
            self.valid_eval = tf.nn.softmax(self.valid_layers[-1])
            self.test_eval = tf.nn.softmax(self.test_layers[-1])

            self.train_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch, logits=self.train_layers[-1])
            self.valid_loss = tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch, logits=self.valid_layers[-1])
            self.test_loss = tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch, logits=self.test_layers[-1])
        else:
            self.train_eval = self.train_layers[-1]
            self.valid_eval = self.valid_layers[-1]
            self.test_eval = self.test_layers[-1]

            self.train_loss = 2.0* tf.nn.l2_loss(self.train_eval-train_y_batch)
            self.valid_loss = 2.0* tf.nn.l2_loss(self.valid_eval-valid_y_batch)
            self.test_loss = 2.0* tf.nn.l2_loss(self.test_eval-test_y_batch)

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss)


########################################################
#### Single Feedforward Net for Multi-task Learning ####
########################################################
#### FFNN3 model for MTL
class MTL_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
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

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.train_eval, self.param = [], [], []
        for task_cnt in range(self.num_tasks):
            layers_of_model = []
            for layer_cnt in range(self.num_layers):
                if task_cnt == 0 and layer_cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(train_x_batch[task_cnt], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1])
                    self.param = self.param + para_tmp
                elif task_cnt == 0 and layer_cnt == self.num_layers-1:
                    layer_tmp, para_tmp = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], activation_fn=None)
                    self.param = self.param + para_tmp
                elif task_cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1])
                    self.param = self.param + para_tmp
                elif layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(train_x_batch[task_cnt], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                elif layer_cnt == self.num_layers-1:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], activation_fn=None, weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])                    
                else:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layers_of_model.append(layer_tmp)
            self.train_models.append(layers_of_model)
            #self.train_eval.append(self.train_models[-1][-1])

        #### layers of model for validation data
        self.valid_models, self.valid_eval = [], []
        for task_cnt in range(self.num_tasks):
            layers_of_model = []
            for layer_cnt in range(self.num_layers):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(valid_x_batch[task_cnt], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                elif layer_cnt == self.num_layers-1:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], activation_fn=None, weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                else:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layers_of_model.append(layer_tmp)
            self.valid_models.append(layers_of_model)
            #self.valid_eval.append(self.valid_models[-1][-1])

        #### layers of model for test data
        self.test_models, self.test_eval = [], []
        for task_cnt in range(self.num_tasks):
            layers_of_model = []
            for layer_cnt in range(self.num_layers):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(test_x_batch[task_cnt], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                elif layer_cnt == self.num_layers-1:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], activation_fn=None, weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                else:
                    layer_tmp, _ = new_fc_layer(layers_of_model[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layers_of_model.append(layer_tmp)
            self.test_models.append(layers_of_model)
            #self.test_eval.append(self.test_models[-1][-1])

        #### functions of model
        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[x][-1]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[x][-1]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[x][-1]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[x][-1]) for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[x][-1]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[x][-1]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[x][-1] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[x][-1] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[x][-1] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

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
#### Single Feedforward Net for Multi-task Learning ####
########         Hard Parameter Sharing         ########
########################################################
class MTL_FFNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_layers, batch_size, learning_rate, learning_rate_decay, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = [len(dim_layers[0])-1, (len(dim_layers[1])//self.num_tasks)]
        self.shared_layers_size = dim_layers[0]
        self.independent_layers_size = dim_layers[1]
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

        #### mini-batch for training/validation/test data
        train_x_batch = [tf.slice(self.train_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        train_y_batch = [tf.slice(self.train_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_x_batch = [tf.slice(self.valid_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        valid_y_batch = [tf.slice(self.valid_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_x_batch = [tf.slice(self.test_x[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]
        test_y_batch = [tf.slice(self.test_y[x], [self.batch_size * self.data_index, 0], [self.batch_size, -1]) for x in range(self.num_tasks)]

        #### layers of model for train data
        self.train_models, self.param = [], []
        for layer_cnt in range(self.num_layers[0]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if task_cnt == 0 and layer_cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(train_x_batch[task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1])
                    self.param = self.param + para_tmp
                elif layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(train_x_batch[task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                elif task_cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(self.train_models[layer_cnt-1][task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1])
                    self.param = self.param + para_tmp
                else:
                    layer_tmp, _ = new_fc_layer(self.train_models[layer_cnt-1][task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layer_of_model.append(layer_tmp)
            self.train_models.append(layer_of_model)

        for layer_cnt in range(self.num_layers[1]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if layer_cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(self.train_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.shared_layers_size[-1], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt])
                elif layer_cnt == self.num_layers[1]-1:
                    layer_tmp, para_tmp = new_fc_layer(self.train_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], activation_fn=None)
                else:
                    layer_tmp, para_tmp = new_fc_layer(self.train_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt])
                layer_of_model.append(layer_tmp)
                self.param = self.param+para_tmp
            self.train_models.append(layer_of_model)
        #self.train_eval = self.train_models[-1]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers[0]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(valid_x_batch[task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                else:
                    layer_tmp, _ = new_fc_layer(self.valid_models[layer_cnt-1][task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layer_of_model.append(layer_tmp)
            self.valid_models.append(layer_of_model)

        for layer_cnt in range(self.num_layers[1]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(self.valid_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.shared_layers_size[-1], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                elif layer_cnt == self.num_layers[1]-1:
                    layer_tmp, _ = new_fc_layer(self.valid_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], activation_fn=None, weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                else:
                    layer_tmp, _ = new_fc_layer(self.valid_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                layer_of_model.append(layer_tmp)
            self.valid_models.append(layer_of_model)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers[0]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(test_x_batch[task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                else:
                    layer_tmp, _ = new_fc_layer(self.test_models[layer_cnt-1][task_cnt], self.shared_layers_size[layer_cnt], self.shared_layers_size[layer_cnt+1], weight=self.param[2*layer_cnt], bias=self.param[2*layer_cnt+1])
                layer_of_model.append(layer_tmp)
            self.test_models.append(layer_of_model)

        for layer_cnt in range(self.num_layers[1]):
            layer_of_model = []
            for task_cnt in range(self.num_tasks):
                if layer_cnt == 0:
                    layer_tmp, _ = new_fc_layer(self.test_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.shared_layers_size[-1], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                elif layer_cnt == self.num_layers[1]-1:
                    layer_tmp, _ = new_fc_layer(self.test_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], activation_fn=None, weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                else:
                    layer_tmp, _ = new_fc_layer(self.test_models[self.num_layers[0]+layer_cnt-1][task_cnt], self.independent_layers_size[(layer_cnt-1)*self.num_tasks+task_cnt], self.independent_layers_size[layer_cnt*self.num_tasks+task_cnt], weight=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)], bias=self.param[2*(self.num_layers[0]+self.num_tasks*layer_cnt+task_cnt)+1])
                layer_of_model.append(layer_tmp)
            self.test_models.append(layer_of_model)
        #self.test_eval = self.test_models[-1]

        #### functions of model
        '''
        self.train_loss_list = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
        self.valid_loss_list = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
        self.test_loss_list = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.train_loss_all = tf.add_n(self.train_loss_list)
        self.valid_loss_all = tf.add_n(self.valid_loss_list)
        self.test_loss_all = tf.add_n(self.test_loss_list)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_list[x]) for x in range(self.num_tasks)]
        self.update_all = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_all)
        '''

        if classification and self.independent_layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.independent_layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]


########################################################
######   Feedforward Net for Multi-task Learning   #####
######            Tensor Factorization             #####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB_W0, KB_b0, KB_W1, KB_b1, ...]
class MTL_FFNN_tensorfactor_minibatch():
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
                layer_tmp, KB_para_tmp, TS_para_tmp = new_tensorfactor_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_tensorfactor_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_tensorfactor_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_tensorfactor_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_tensorfactor_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_tensorfactor_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_tensorfactor_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_tensorfactor_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_tensorfactor_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        '''
        self.train_loss_list = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
        self.train_loss_and_reg_list = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) + reg_term1 + reg_term2 for x in range(self.num_tasks)]
        self.valid_loss_list = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
        self.test_loss_list = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.train_loss_all = tf.add_n(self.train_loss_list)
        self.train_loss_and_reg_all = self.train_loss_all + reg_term1 + reg_term2
        self.valid_loss_all = tf.add_n(self.valid_loss_list)
        self.test_loss_all = tf.add_n(self.test_loss_list)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg_list[x]) for x in range(self.num_tasks)]
        self.update_all = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg_all)
        '''

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

