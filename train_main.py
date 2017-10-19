import os
import pickle
from time import sleep
import timeit
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.io import savemat, loadmat

from gen_data import sine_data_print_info, mnist_data_print_info

from ffnn_baseline_model import FFNN_batch, FFNN_minibatch, MTL_FFNN_minibatch, MTL_FFNN_HPS_minibatch, MTL_FFNN_tensorfactor_minibatch
from ffnn_ella_model import ELLA_FFNN_simple_minibatch, ELLA_FFNN_linear_relation_minibatch, ELLA_FFNN_linear_relation_minibatch2, ELLA_FFNN_nonlinear_relation_minibatch, ELLA_FFNN_nonlinear_relation_minibatch2
from cnn_baseline_model import CNN_batch, CNN_minibatch, MTL_CNN_minibatch
from cnn_ella_model import ELLA_CNN_relation2_minibatch, ELLA_CNN_deconv_relation_minibatch

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info, classification_prob=False, data_list=None):
    learning_model, gen_model_success = None, True
    learning_rate = train_hyperpara['lr']
    learning_rate_decay = train_hyperpara['lr_decay']

    if classification_prob:
        if len(data_info) == 3:
            x_dim, y_dim, y_depth = data_info
        elif len(data_info) == 4:
            x_dim, y_dim, y_depth, num_task = data_info
        layers_dimension = [x_dim] + model_hyperpara['hidden_layer'] + [y_depth]
        fc_hidden_size = model_hyperpara['hidden_layer'] + [y_depth]
    else:
        if len(data_info) == 2:
            x_dim, y_dim = data_info
        elif len(data_info) == 3:
            x_dim, y_dim, num_task = data_info
        layers_dimension = [x_dim] + model_hyperpara['hidden_layer'] + [y_dim]

    ###### FFNN models
    if model_architecture is 'ffnn_batch':
        print("Training batch FFNN model")
        learning_model = FFNN_batch(dim_layers=layers_dimension, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, classification=classification_prob)
    elif model_architecture is 'ffnn_minibatch':
        print("Training mini-batch FFNN model")
        batch_size = model_hyperpara['batch_size']
        learning_model = FFNN_minibatch(dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'mtl_ffnn_minibatch':
        print("Training MTL-FFNN model (Single NN ver.)")
        batch_size = model_hyperpara['batch_size']
        learning_model = MTL_FFNN_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'mtl_ffnn_hard_para_sharing':
        print("Training MTL-FFNN model (Hard Parameter Sharing Ver.)")
        ts_layer_dim_tmp = model_hyperpara['task_specific_layer']
        if classification_prob:
            layers_dimension = [[x_dim]+model_hyperpara['hidden_layer'], [ts_layer_dim_tmp[x]+[max_y] for x in range(num_task)]]
        else:
            layers_dimension = [[x_dim]+model_hyperpara['hidden_layer'], [ts_layer_dim_tmp[x]+[y_dim] for x in range(num_task)]]
        batch_size = model_hyperpara['batch_size']
        learning_model = MTL_FFNN_HPS_minibatch(num_tasks=num_task, dim_layers=layers_dimension, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'mtl_ffnn_tensorfactor':
        print("Training MTL-FFNN model (ver. tensor factorization)")
        know_base_dimension = model_hyperpara['knowledge_base']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = MTL_FFNN_tensorfactor_minibatch(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'ELLA_ffnn_simple':
        print("Training ELLA-FFNN model (ver. simple)")
        know_base_dimension = model_hyperpara['knowledge_base']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = ELLA_FFNN_simple_minibatch(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'ELLA_ffnn_linear_relation':
        print("Training ELLA-FFNN model (ver. linear relation from KB to TS)")
        know_base_dimension = model_hyperpara['knowledge_base']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = ELLA_FFNN_linear_relation_minibatch(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'ELLA_ffnn_linear_relation2':
        print("Training ELLA-FFNN model (ver. 2-layer linear relation from KB to TS)")
        know_base_dimension = model_hyperpara['knowledge_base']
        task_specific_dimension = model_hyperpara['task_specific']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = ELLA_FFNN_linear_relation_minibatch2(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, dim_task_specific=task_specific_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'ELLA_ffnn_nonlinear_relation':
        print("Training ELLA-FFNN model (ver. NON-linear relation from KB to TS)")
        know_base_dimension = model_hyperpara['knowledge_base']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = ELLA_FFNN_nonlinear_relation_minibatch(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    elif model_architecture is 'ELLA_ffnn_nonlinear_relation2':
        print("Training ELLA-FFNN model (ver. 2-layer NON-linear relation from KB to TS)")
        know_base_dimension = model_hyperpara['knowledge_base']
        task_specific_dimension = model_hyperpara['task_specific']
        batch_size = model_hyperpara['batch_size']
        regularization_scale = model_hyperpara['regularization_scale']
        learning_model = ELLA_FFNN_nonlinear_relation_minibatch2(num_tasks=num_task, dim_layers=layers_dimension, dim_know_base=know_base_dimension, dim_task_specific=task_specific_dimension, reg_scale=regularization_scale, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, classification=classification_prob)
    ###### CNN models
    elif model_architecture is 'cnn_batch':
        print("Training batch CNN-FC model")
        cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
        cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
        if cnn_pooling:
            cnn_pool_size = model_hyperpara['pooling_size']
        else:
            cnn_pool_size = None
        input_img_size = model_hyperpara['image_dimension']
        learning_model = CNN_batch(dim_channels=cnn_channel_size, dim_fcs=fc_hidden_size, dim_img=input_img_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif model_architecture is 'cnn_minibatch':
        print("Training mini-batch CNN-FC model")
        cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
        cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
        if cnn_pooling:
            cnn_pool_size = model_hyperpara['pooling_size']
        else:
            cnn_pool_size = None
        batch_size = model_hyperpara['batch_size']
        input_img_size = model_hyperpara['image_dimension']
        learning_model = CNN_minibatch(dim_channels=cnn_channel_size, dim_fcs=fc_hidden_size, dim_img=input_img_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif model_architecture is 'mtl_cnn_minibatch':
        print("Training MTL-CNN model (Single NN ver.)")
        cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
        cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
        if cnn_pooling:
            cnn_pool_size = model_hyperpara['pooling_size']
        else:
            cnn_pool_size = None
        batch_size = model_hyperpara['batch_size']
        input_img_size = model_hyperpara['image_dimension']
        learning_model = MTL_CNN_minibatch(dim_channels=cnn_channel_size, num_tasks=num_task, dim_fcs=fc_hidden_size, dim_img=input_img_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, data_list=data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif ('ELLA_cnn' in model_architecture and 'linear_relation2' in model_architecture):
        cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
        cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
        if cnn_pooling:
            cnn_pool_size = model_hyperpara['pooling_size']
        else:
            cnn_pool_size = None
        batch_size = model_hyperpara['batch_size']
        input_img_size = model_hyperpara['image_dimension']

        cnn_know_base_size, fc_know_base_size = model_hyperpara['cnn_KB_sizes'], model_hyperpara['fc_KB_sizes']
        regularization_scale = model_hyperpara['regularization_scale']
        if model_architecture is 'ELLA_cnn_linear_relation2':
            print("Training ELLA-CNN model (ver. 2-layer linear relation from KB to TS)")
            learning_model = ELLA_CNN_relation2_minibatch(num_task, cnn_channel_size, fc_hidden_size, input_img_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, fc_know_base_size, batch_size, learning_rate, learning_rate_decay, regularization_scale, data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn=None)
        elif model_architecture is 'ELLA_cnn_nonlinear_relation2':
            print("Training ELLA-CNN model (ver. 2-layer NON-linear relation from KB to TS)")
            learning_model = ELLA_CNN_relation2_minibatch(num_task, cnn_channel_size, fc_hidden_size, input_img_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, fc_know_base_size, batch_size, learning_rate, learning_rate_decay, regularization_scale, data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn=tf.nn.relu)
    elif ('ELLA_cnn_deconv_relation' in model_architecture):
        cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
        cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
        if cnn_pooling:
            cnn_pool_size = model_hyperpara['pooling_size']
        else:
            cnn_pool_size = None
        batch_size = model_hyperpara['batch_size']
        input_img_size = model_hyperpara['image_dimension']

        cnn_know_base_size, fc_know_base_size = model_hyperpara['cnn_KB_sizes'], model_hyperpara['fc_KB_sizes']
        cnn_task_specific_size, cnn_deconv_stride_size = model_hyperpara['cnn_TS_sizes'], model_hyperpara['cnn_deconv_stride_sizes']
        regularization_scale = model_hyperpara['regularization_scale']
        if model_architecture is 'ELLA_cnn_deconv_relation_relu':
            print("Training ELLA-CNN_Deconv model (ReLu act at Deconv)")
            learning_model = ELLA_CNN_deconv_relation_minibatch(num_task, cnn_channel_size, fc_hidden_size, input_img_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size, fc_know_base_size, batch_size, learning_rate, learning_rate_decay, regularization_scale, data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn_cnn=tf.nn.relu, relation_activation_fn_fc=tf.nn.relu)
        elif model_architecture is 'ELLA_cnn_deconv_relation_tanh':
            print("Training ELLA-CNN_Deconv model (tanh act at Deconv)")
            learning_model = ELLA_CNN_deconv_relation_minibatch(num_task, cnn_channel_size, fc_hidden_size, input_img_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size, fc_know_base_size, batch_size, learning_rate, learning_rate_decay, regularization_scale, data_list, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn_cnn=tf.nn.tanh, relation_activation_fn_fc=tf.nn.relu)
    else:
        print("No such model exists!!")
        print("No such model exists!!")
        print("No such model exists!!")
        gen_model_success = False
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train_main(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_result=False, result_folder_name=None):
    ### control log of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    tf.logging.set_verbosity(tf.logging.ERROR)

    if useGPU:
        config = tf.ConfigProto(device_count = {'GPU': GPU_device})
        #config = tf.ConfigProto(device_count = {'GPU': GPU_device}, log_device_placement=True)
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        print("GPU %d is used" %(GPU_device))
    else:
        config = tf.ConfigProto()
        print("CPU is used")
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    ### set-up data
    train_data, validation_data, test_data = dataset
    if data_type is 'sine':
        num_task, num_train, num_valid, num_test, x_dim, y_dim = sine_data_print_info(train_data, validation_data, test_data, print_info=False)
    elif data_type is 'sine_plus_linear':
        num_task, num_train, num_valid, num_test, x_dim, y_dim = sine_data_print_info(train_data, validation_data, test_data, print_info=False)
    elif data_type is 'mnist':
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']


    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], classification_prob=classification_prob, data_list=dataset)
    if not generation_success:
        return (None, None, None, None)


    ### Training Procedure
    if save_result:
        best_para_file_name = './'+result_folder_name+'/best_model_parameter.pkl'

    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [range(num_train[x]//batch_size) for x in range(num_task)]
        else:
            indices = [range(num_train//batch_size)]

    best_valid_error, test_error_at_best_epoch, best_epoch = np.inf, np.inf, -1
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []
    best_param = []

    if not save_result:
        print("Not saving result")
    elif result_folder_name in os.listdir(os.getcwd()):
        print("Subfolder exists")
    else:
        print("Not Exist, so make that")
        os.mkdir(result_folder_name)
    #sleep(2)

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)

        #tfboard_writer = tf.summary.FileWriter('./graphs', sess.graph)
        #return None


        start_time = timeit.default_timer()
        while learning_step < min(learning_step_max, patience):
            learning_step = learning_step+1

            #### training & performance measuring process
            if model_architecture is 'ffnn_batch':
                #### batch FFNN for single task
                if learning_step > 0:
                    sess.run(learning_model.update, feed_dict={learning_model.model_input:train_x, learning_model.true_output:train_y, learning_model.epoch:learning_step-1})

                if classification_prob:
                    #### it is accuracy, not error
                    train_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:train_x, learning_model.true_output:train_y})

                    validation_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:valid_x, learning_model.true_output:valid_y})

                    test_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:test_x, learning_model.true_output:test_y})

                    train_error, valid_error, test_error = -train_error_tmp/num_train, -validation_error_tmp/num_valid, -test_error_tmp/num_test
                else:
                    train_error_tmp = sess.run(learning_model.loss, feed_dict={learning_model.model_input:train_x, learning_model.true_output:train_y})

                    validation_error_tmp = sess.run(learning_model.loss, feed_dict={learning_model.model_input:valid_x, learning_model.true_output:valid_y})

                    test_error_tmp = sess.run(learning_model.loss, feed_dict={learning_model.model_input:test_x, learning_model.true_output:test_y})

                    train_error, valid_error, test_error = np.sqrt(train_error_tmp/num_train), np.sqrt(validation_error_tmp/num_valid), np.sqrt(test_error_tmp/num_test)

            elif model_architecture is 'ffnn_minibatch':
                #### mini-batch FFNN for single task
                shuffle(indices[0])
                if learning_step > 0:
                    for batch_cnt in range(num_train//batch_size):
                        sess.run(learning_model.update, feed_dict={learning_model.data_index:indices[0][batch_cnt], learning_model.epoch:learning_step-1})

                if classification_prob:
                    #### it is accuracy, not error
                    train_error_tmp = 0.0
                    for batch_cnt in range(num_train//batch_size):
                        train_error_tmp = train_error_tmp + sess.run(learning_model.train_accuracy, feed_dict={learning_model.data_index:batch_cnt})

                    validation_error_tmp = 0.0
                    for batch_cnt in range(num_valid//batch_size):
                        validation_error_tmp = validation_error_tmp + sess.run(learning_model.valid_accuracy, feed_dict={learning_model.data_index:batch_cnt})

                    test_error_tmp = 0.0
                    for batch_cnt in range(num_test//batch_size):
                        test_error_tmp = test_error_tmp + sess.run(learning_model.test_accuracy, feed_dict={learning_model.data_index:batch_cnt})

                    train_error, valid_error, test_error = -train_error_tmp/num_train, -validation_error_tmp/num_valid, -test_error_tmp/num_test
                else:
                    train_error_tmp = 0.0
                    for batch_cnt in range(num_train//batch_size):
                        train_error_tmp = train_error_tmp + sess.run(learning_model.train_loss, feed_dict={learning_model.data_index:batch_cnt})

                    validation_error_tmp = 0.0
                    for batch_cnt in range(num_valid//batch_size):
                        validation_error_tmp = validation_error_tmp + sess.run(learning_model.valid_loss, feed_dict={learning_model.data_index:batch_cnt})

                    test_error_tmp = 0.0
                    for batch_cnt in range(num_test//batch_size):
                        test_error_tmp = test_error_tmp + sess.run(learning_model.test_loss, feed_dict={learning_model.data_index:batch_cnt})

                    train_error, valid_error, test_error = np.sqrt(train_error_tmp/num_train), np.sqrt(validation_error_tmp/num_valid), np.sqrt(test_error_tmp/num_test)

            elif model_architecture is 'cnn_batch':
                #### batch CNN for single task
                if learning_step > 0:
                    sess.run(learning_model.update, feed_dict={learning_model.model_input:train_x, learning_model.true_output:train_y, learning_model.epoch:learning_step-1, learning_model.dropout_prob:0.5})

                train_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:train_x, learning_model.true_output:train_y, learning_model.dropout_prob:1.0})

                validation_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:valid_x, learning_model.true_output:valid_y, learning_model.dropout_prob:1.0})

                test_error_tmp = sess.run(learning_model.accuracy, feed_dict={learning_model.model_input:test_x, learning_model.true_output:test_y, learning_model.dropout_prob:1.0})

                train_error, valid_error, test_error = -train_error_tmp/num_train, -validation_error_tmp/num_valid, -test_error_tmp/num_test

            elif model_architecture is 'cnn_minibatch':
                #### mini-batch CNN for single task
                shuffle(indices)
                if learning_step > 0:
                    for batch_cnt in range(num_train//batch_size):
                        sess.run(learning_model.update, feed_dict={learning_model.data_index:indices[0][batch_cnt], learning_model.epoch:learning_step-1, learning_model.dropout_prob:0.5})

                train_error_tmp = 0.0
                for batch_cnt in range(num_train//batch_size):
                    train_error_tmp = train_error_tmp + sess.run(learning_model.train_accuracy, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})

                validation_error_tmp = 0.0
                for batch_cnt in range(num_valid//batch_size):
                    validation_error_tmp = validation_error_tmp + sess.run(learning_model.valid_accuracy, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})

                test_error_tmp = 0.0
                for batch_cnt in range(num_test//batch_size):
                    test_error_tmp = test_error_tmp + sess.run(learning_model.test_accuracy, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})

                train_error, valid_error, test_error = -train_error_tmp/num_train, -validation_error_tmp/num_valid, -test_error_tmp/num_test

            elif (model_architecture is 'mtl_ffnn_minibatch') or (model_architecture is 'mtl_ffnn_hard_para_sharing') or (model_architecture is 'mtl_ffnn_tensorfactor') or (model_architecture is 'ELLA_ffnn_simple') or (model_architecture is 'ELLA_ffnn_linear_relation') or (model_architecture is 'ELLA_ffnn_linear_relation2') or (model_architecture is 'ELLA_ffnn_nonlinear_relation') or (model_architecture is 'ELLA_ffnn_nonlinear_relation2') or (model_architecture is 'mtl_cnn_minibatch') or (model_architecture is 'ELLA_cnn_linear_relation2') or (model_architecture is 'ELLA_cnn_nonlinear_relation2') or ('ELLA_cnn_deconv' in model_architecture):
                #### Multi-task models
                task_for_train = np.random.randint(0, num_task)
                shuffle(indices[task_for_train])
                if learning_step > 0:
                    for batch_cnt in range(num_train[task_for_train]//batch_size):
                        if 'cnn' in model_architecture:
                            sess.run(learning_model.update[task_for_train], feed_dict={learning_model.data_index:indices[task_for_train][batch_cnt], learning_model.epoch:learning_step-1, learning_model.dropout_prob:0.6})
                        else:
                            sess.run(learning_model.update[task_for_train], feed_dict={learning_model.data_index:indices[task_for_train][batch_cnt], learning_model.epoch:learning_step-1})

                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for task_cnt in range(num_task):
                    if classification_prob:
                        model_train_error, model_valid_error, model_test_error = learning_model.train_accuracy[task_cnt], learning_model.valid_accuracy[task_cnt], learning_model.test_accuracy[task_cnt]
                    else:
                        model_train_error, model_valid_error, model_test_error = learning_model.train_loss[task_cnt], learning_model.valid_loss[task_cnt], learning_model.test_loss[task_cnt]

                    for batch_cnt in range(num_train[task_cnt]//batch_size):
                        if 'cnn' in model_architecture:
                            train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})
                        else:
                            train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error, feed_dict={learning_model.data_index:batch_cnt})
                    train_error_tmp[task_cnt] = train_error_tmp[task_cnt]/num_train[task_cnt]

                    for batch_cnt in range(num_valid[task_cnt]//batch_size):
                        if 'cnn' in model_architecture:
                            validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})
                        else:
                            validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error, feed_dict={learning_model.data_index:batch_cnt})
                    validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt]/num_valid[task_cnt]

                    for batch_cnt in range(num_test[task_cnt]//batch_size):
                        if 'cnn' in model_architecture:
                            test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error, feed_dict={learning_model.data_index:batch_cnt, learning_model.dropout_prob:1.0})
                        else:
                            test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error, feed_dict={learning_model.data_index:batch_cnt})
                    test_error_tmp[task_cnt] = test_error_tmp[task_cnt]/num_test[task_cnt]

                if classification_prob:
                    train_error, valid_error, test_error = -(sum(train_error_tmp)/num_task), -(sum(validation_error_tmp)/num_task), -(sum(test_error_tmp)/num_task)
                else:
                    train_error, valid_error, test_error = np.sqrt(train_error_tmp/num_task), np.sqrt(validation_error_tmp/num_task), np.sqrt(test_error_tmp/num_task)

            #### current parameter of model
            curr_param = sess.run(learning_model.param)

            #### error related process
            print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error), abs(valid_error)))

            if valid_error < best_valid_error:
                if valid_error < best_valid_error * improvement_threshold:
                    patience = max(patience, learning_step*patience_multiplier)
                best_valid_error, best_epoch = valid_error, learning_step
                test_error_at_best_epoch = test_error
                print('\t\t\t\t\t\t\tTest : %f' % (abs(test_error_at_best_epoch)))

                #### save best parameter of model
                if save_result:
                    best_param = sess.run(learning_model.param)
                    with open(best_para_file_name, 'wb') as best_para_fobj:
                        pickle.dump([best_param, test_error_at_best_epoch, best_epoch], best_para_fobj)
                        print('\t\tSave best parameter')

            train_error_hist.append(abs(train_error))
            valid_error_hist.append(abs(valid_error))
            test_error_hist.append(abs(test_error))
            best_test_error_hist.append(abs(test_error_at_best_epoch))

            #### save intermediate result of training procedure
            if (learning_step % 50 == 0) and save_result:
                para_file_name = './' + result_folder_name + '/model_parameter(epoch_' + str(learning_step) + ').pkl'
                with open(para_file_name, 'wb') as para_fobj:
                    pickle.dump([curr_param, best_param], para_fobj)

                print('\t\tsave summary of training')

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    print("Best validation error : %.4f (at epoch %d)" %(abs(best_valid_error), best_epoch))
    print("Test error at that epoch (%d) : %.4f" %(best_epoch, abs(test_error_at_best_epoch)))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist
    result_summary['best_validation_error'] = abs(best_valid_error)
    result_summary['test_error_at_best_epoch'] = abs(test_error_at_best_epoch)

    #tfboard_writer.close()

    return result_summary
