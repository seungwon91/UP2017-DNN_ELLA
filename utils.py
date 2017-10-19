import numpy as np
import tensorflow as tf
from math import ceil, sqrt

############################################
#####     functions for saving data    #####
############################################
def tflized_data(data_list, do_MTL, num_tasks=0):
    with tf.name_scope('RawData_Input'):
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
    with tf.name_scope('Minibatch_Data'):
        if not do_MTL:
            #### single task
            train_x_batch = tf.slice(data_list[0], [batch_size * data_index, 0], [batch_size, -1])
            valid_x_batch = tf.slice(data_list[2], [batch_size * data_index, 0], [batch_size, -1])
            test_x_batch = tf.slice(data_list[4], [batch_size * data_index, 0], [batch_size, -1])

            if len(data_list[1].shape) == 1:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index], [batch_size])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index], [batch_size])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index], [batch_size])
            else:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index, 0], [batch_size, -1])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index, 0], [batch_size, -1])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index, 0], [batch_size, -1])
        else:
            #### multi-task
            train_x_batch = [tf.slice(data_list[0][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
            valid_x_batch = [tf.slice(data_list[2][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
            test_x_batch = [tf.slice(data_list[4][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]

            if len(data_list[1][0].shape) == 1:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
            else:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


def minibatched_cnn_data(data_list, batch_size, data_index, data_tensor_dim, do_MTL, num_tasks=0):
    with tf.name_scope('Minibatch_Data_CNN'):
        if not do_MTL:
            #### single task
            train_x_batch = tf.reshape(tf.slice(data_list[0], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
            valid_x_batch = tf.reshape(tf.slice(data_list[2], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
            test_x_batch = tf.reshape(tf.slice(data_list[4], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)

            if len(data_list[1].shape) == 1:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index], [batch_size])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index], [batch_size])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index], [batch_size])
            else:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index, 0], [batch_size, -1])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index, 0], [batch_size, -1])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index, 0], [batch_size, -1])
        else:
            #### multi-task
            train_x_batch = [tf.reshape(tf.slice(data_list[0][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
            valid_x_batch = [tf.reshape(tf.slice(data_list[2][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
            test_x_batch = [tf.reshape(tf.slice(data_list[4][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]

            if len(data_list[1][0].shape) == 1:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
            else:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


############################################
#### functions for (MTL) model's output ####
############################################
def mtl_model_output_functions(models, y_batches, num_tasks, dim_output, classification=False):
    if classification:
        with tf.name_scope('Model_Eval'):
            train_eval = [tf.nn.softmax(models[0][x][-1]) for x in range(num_tasks)]
            valid_eval = [tf.nn.softmax(models[1][x][-1]) for x in range(num_tasks)]
            test_eval = [tf.nn.softmax(models[2][x][-1]) for x in range(num_tasks)]

        #dim_output = models[0][0][-1].get_shape()[1]
        one_hot_y_batches = []
        one_hot_y_batches.append([tf.one_hot(indices=tf.cast(y_batches[0][x], tf.int32), depth=dim_output) for x in range(num_tasks)])
        one_hot_y_batches.append([tf.one_hot(indices=tf.cast(y_batches[1][x], tf.int32), depth=dim_output) for x in range(num_tasks)])
        one_hot_y_batches.append([tf.one_hot(indices=tf.cast(y_batches[2][x], tf.int32), depth=dim_output) for x in range(num_tasks)])

        with tf.name_scope('Model_Loss'):
            train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_batches[0][x], logits=models[0][x][-1]) for x in range(num_tasks)]
            valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_batches[1][x], logits=models[1][x][-1]) for x in range(num_tasks)]
            test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_batches[2][x], logits=models[2][x][-1]) for x in range(num_tasks)]

        with tf.name_scope('Model_Accuracy'):
            train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(train_eval[x], 1), tf.argmax(one_hot_y_batches[0][x], 1)), tf.float32)) for x in range(num_tasks)]
            valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(valid_eval[x], 1), tf.argmax(one_hot_y_batches[1][x], 1)), tf.float32)) for x in range(num_tasks)]
            test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test_eval[x], 1), tf.argmax(one_hot_y_batches[2][x], 1)), tf.float32)) for x in range(num_tasks)]
    else:
        with tf.name_scope('Model_Eval'):
            train_eval = [models[0][x][-1] for x in range(num_tasks)]
            valid_eval = [models[1][x][-1] for x in range(num_tasks)]
            test_eval = [models[2][x][-1] for x in range(num_tasks)]

        with tf.name_scope('Model_Loss'):
            train_loss = [2.0* tf.nn.l2_loss(train_eval[x]-y_batches[0][x]) for x in range(num_tasks)]
            valid_loss = [2.0* tf.nn.l2_loss(valid_eval[x]-y_batches[1][x]) for x in range(num_tasks)]
            test_loss = [2.0* tf.nn.l2_loss(test_eval[x]-y_batches[2][x]) for x in range(num_tasks)]

        train_accuracy, valid_accuracy, test_accuracy = None, None, None
    return (train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy)


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



############################################
#### functions to save training results ####
############################################
def mean_of_list(list_input):
    return float(sum(list_input))/len(list_input)

def stddev_of_list(list_input):
    list_mean = mean_of_list(list_input)
    sq_err = [(x-list_mean)**2 for x in list_input]

    if len(list_input)<2:
        return 0.0
    else:
        return sqrt(sum(sq_err)/float(len(list_input)-1))


def model_info_summary(model_architecture, model_hyperpara, train_hyperpara):
    tmp_dict = {}
    tmp_dict['architecture'] = model_architecture
    tmp_dict['learning_rate'] = train_hyperpara['lr']
    tmp_dict['improvement_threshold'] = train_hyperpara['improvement_threshold']
    tmp_dict['early_stopping_para'] = [train_hyperpara['patience'], train_hyperpara['patience_multiplier']]

    tmp_dict['batch_size'] = model_hyperpara['batch_size']
    tmp_dict['hidden_layer'] = model_hyperpara['hidden_layer']
    if model_architecture is 'mtl_ffnn_hard_para_sharing':
        tmp_dict['task_specific_layer'] = model_hyperpara['task_specific_layer']
    if ('mtl' in model_architecture and 'tensorfactor' in model_architecture) or ('ELLA_ffnn' in model_architecture and ('simple' in model_architecture or 'relation' in model_architecture)):
        tmp_dict['knowledge_base'] = model_hyperpara['knowledge_base']
        tmp_dict['regularization_scale'] = model_hyperpara['regularization_scale']
    if ('ELLA_ffnn' in model_architecture and 'relation2' in model_architecture):
        tmp_dict['task_specific'] = model_hyperpara['task_specific']
    if ('cnn' in model_architecture):
        tmp_dict['kernel_sizes'] = model_hyperpara['kernel_sizes']
        tmp_dict['stride_sizes'] = model_hyperpara['stride_sizes']
        tmp_dict['channel_sizes'] = model_hyperpara['channel_sizes']
        tmp_dict['padding_type'] = model_hyperpara['padding_type']
        tmp_dict['max_pooling'] = model_hyperpara['max_pooling']
        tmp_dict['pooling_size'] = model_hyperpara['pooling_size']
        tmp_dict['dropout'] = model_hyperpara['dropout']
        tmp_dict['image_dimension'] = model_hyperpara['image_dimension']
    if ('ELLA_cnn' in model_architecture and 'linear_relation2' in model_architecture):
        tmp_dict['regularization_scale'] = model_hyperpara['regularization_scale']
        tmp_dict['cnn_knowledge_base'] = model_hyperpara['cnn_KB_sizes']
        tmp_dict['fc_knowledge_base'] = model_hyperpara['fc_KB_sizes']
    return tmp_dict


def reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, data_group_list):
    result_of_curr_run = {}
    #### 'model_specific_info' element
    result_of_curr_run['model_specific_info'] = model_info_summary(model_architecture, model_hyperpara, train_hyperpara)

    num_run_per_model, best_valid_error_list, best_test_error_list = train_hyperpara['num_run_per_model'], [], []
    result_of_curr_run['result_of_each_run'] = np.zeros((num_run_per_model,), dtype=np.object)
    for cnt in range(num_run_per_model):
        result_of_curr_run['result_of_each_run'][cnt] = result_from_train_run[cnt]
        best_valid_error_list.append(result_from_train_run[cnt]['best_validation_error'])
        best_test_error_list.append(result_from_train_run[cnt]['test_error_at_best_epoch'])

    result_of_curr_run['best_valid_error'] = best_valid_error_list
    result_of_curr_run['best_valid_error_mean'] = mean_of_list(best_valid_error_list)
    result_of_curr_run['best_valid_error_stddev'] = stddev_of_list(best_valid_error_list)
    result_of_curr_run['best_test_error'] = best_test_error_list
    result_of_curr_run['best_test_error_mean'] = mean_of_list(best_test_error_list)
    result_of_curr_run['best_test_error_stddev'] = stddev_of_list(best_test_error_list)
    result_of_curr_run['train_valid_data_group'] = data_group_list

    return result_of_curr_run
