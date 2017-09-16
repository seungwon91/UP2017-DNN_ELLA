import numpy as np
import tensorflow as tf
from misc_functions_for_model import *

#################################################
############ Simple Feedforward Net #############
#################################################
#### Feedforward Neural Net
class FFNN_batch():
    def __init__(self, dim_layers, learning_rate, learning_rate_decay, classification=False):
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay

        #### placeholder of model
        self.model_input = new_placeholder([None, self.layers_size[0]])
        self.true_output = new_placeholder([None, self.layers_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        if classification:
            self.layers, self.param = new_fc_net(self.model_input, self.layers_size, params=None, output_type='classification')
        else:
            self.layers, self.param = new_fc_net(self.model_input, self.layers_size, params=None, output_type=None)

        #### functions of model
        if classification and self.layers_size[-1]>1:
            self.eval = tf.nn.softmax(self.layers[-1])
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.true_output, logits=self.layers[-1])
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.eval, 1), tf.argmax(self.true_output, 1)), tf.float32))
        else:
            self.eval = self.layers[-1]
            self.loss = 2.0* tf.nn.l2_loss(self.eval-self.true_output)
            if classification:
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal((self.eval>0.5), (self.true_output>0.5)), tf.float32))

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.loss)


#### Feedforward Neural Network - mini batch ver.
class FFNN_minibatch():
    def __init__(self, dim_layers, batch_size, learning_rate, learning_rate_decay, data_list, classification=False):
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=False)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=False)

        #### layers of model for train data
        if classification:
            self.train_layers, self.param = new_fc_net(train_x_batch, self.layers_size, params=None, output_type='classification')
        else:
            self.train_layers, self.param = new_fc_net(train_x_batch, self.layers_size, params=None, output_type=None)

        #### layers of model for validation data
        if classification:
            self.valid_layers, _ = new_fc_net(valid_x_batch, self.layers_size, params=self.param, output_type='classification')
        else:
            self.valid_layers, _ = new_fc_net(valid_x_batch, self.layers_size, params=self.param, output_type=None)

        #### layers of model for test data
        if classification:
            self.test_layers, _ = new_fc_net(test_x_batch, self.layers_size, params=self.param, output_type='classification')
        else:
            self.test_layers, _ = new_fc_net(test_x_batch, self.layers_size, params=self.param, output_type=None)

        #### functions of model
        if classification and self.layers_size[-1]>1:
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

            if classification:
                self.train_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.train_eval>0.5), (train_y_batch>0.5)), tf.float32))
                self.valid_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.valid_eval>0.5), (valid_y_batch>0.5)), tf.float32))
                self.test_accuracy = tf.reduce_sum(tf.cast(tf.equal((self.test_eval>0.5), (test_y_batch>0.5)), tf.float32))

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
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True, num_tasks=num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0 and classification:
                model_tmp, self.param = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type='classification')
            elif task_cnt == 0:
                model_tmp, self.param = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type=None)
            elif classification:
                model_tmp, _ = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.train_models.append(model_tmp)

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.layers_size[-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]


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
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True, num_tasks=num_tasks)

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
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.layers_size[-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]


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
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True, num_tasks=num_tasks)

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_tensorfactor_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_tensorfactor_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_tensorfactor_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l1_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_tensorfactor_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[2*layer_cnt:2*(layer_cnt+1)], self.num_tasks, layer_cnt, KB_param=self.KB_param[2*layer_cnt:2*(layer_cnt+1)], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.layers_size[-1], classification)
        self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch/self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]
