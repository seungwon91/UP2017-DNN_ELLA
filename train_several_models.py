from os import listdir, getcwd, mkdir
from os.path import isfile
from math import sqrt

from scipy.io import loadmat, savemat
import numpy as np
from tensorflow import reset_default_graph

from gen_data import sine_data, sine_plus_linear_data, mnist_data, mnist_data_print_info
from train_main import train_main


def mean_of_list(list_input):
    return float(sum(list_input))/len(list_input)


def stddev_of_list(list_input):
    list_mean = mean_of_list(list_input)
    sq_err = [(x-list_mean)**2 for x in list_input]
    if len(list_input)<2:
        return 0.0
    else:
        return sqrt(sum(sq_err)/float(len(list_input)-1))

    
def reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, data_group_list):
    result_of_curr_run, tmp_dict = {}, {}
    #### 'model_specific_info' element
    tmp_dict['architecture'] = model_architecture
    tmp_dict['learning_rate'] = train_hyperpara['lr']
    tmp_dict['improvement_threshold'] = train_hyperpara['improvement_threshold']
    tmp_dict['early_stopping_para'] = [train_hyperpara['patience'], train_hyperpara['patience_multiplier']]

    tmp_dict['batch_size'] = model_hyperpara['batch_size']
    tmp_dict['hidden_layer'] = model_hyperpara['hidden_layer']
    if model_architecture is 'mtl_ffnn_hard_para_sharing':
        tmp_dict['task_specific_layer'] = model_hyperpara['task_specific_layer']
    if ('mtl' in model_architecture and 'tensorfactor' in model_architecture) or ('ELLA' in model_architecture and ('simple' in model_architecture or 'relation' in model_architecture)):
        tmp_dict['knowledge_base'] = model_hyperpara['knowledge_base']
        tmp_dict['regularization_scale'] = model_hyperpara['regularization_scale']
    if ('ELLA' in model_architecture and 'relation2' in model_architecture):
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
    result_of_curr_run['model_specific_info'] = tmp_dict

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


def train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, num_data_group, mat_file_name, classification_prob, saved_result=None, useGPU=False, GPU_device=0):
    if not 'Result' in listdir(getcwd()):
        mkdir('Result')

    #### process results of previous training
    if (saved_result is None) and not (isfile('./Result/'+mat_file_name)):
        saved_result = np.zeros((1,), dtype=np.object)
    elif (saved_result is None):
        saved_result_tmp = loadmat('./Result/'+mat_file_name)
        num_prev_test_model = len(saved_result_tmp['training_summary'][0])
        saved_result = np.zeros((num_prev_test_model+1,), dtype=np.object)
        for cnt in range(num_prev_test_model):
            saved_result[cnt] = saved_result_tmp['training_summary'][0][cnt]
    else:
        num_prev_result, prev_result_tmp = len(saved_result), saved_result
        saved_result = np.zeros((num_prev_result+1,), dtype=np.object)
        for cnt in range(num_prev_result):
            saved_result[cnt] = prev_result_tmp[cnt]

    #### run training procedure with different dataset
    max_run_cnt = train_hyperpara['num_run_per_model']
    group_cnt = np.random.randint(0, num_data_group, size=max_run_cnt)
    result_from_train_run = []
    for run_cnt in range(max_run_cnt):
        #group_cnt = np.random.randint(0, num_data_group, size=1)[0]
        train_result_tmp = train_main(model_architecture, model_hyperpara, train_hyperpara, [dataset[0][group_cnt[run_cnt]], dataset[1][group_cnt[run_cnt]], dataset[2]], data_type, classification_prob, useGPU, GPU_device)
        result_from_train_run.append(train_result_tmp)
        print "%d-th training run\n\n" % (run_cnt+1)
        reset_default_graph()

    #### save training summary
    result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, train_data_group_list)
    saved_result[-1] = result_of_curr_run
    savemat('./Result/'+mat_file_name, {'training_summary':saved_result})

    return saved_result



# input arguments : data_type, data_file_name, data_hyperpara, model_architecture, model_hyperpara, train_hyperpara, save_result, result_folder_name=None

# model_architecture : 'ffnn_batch', 'ffnn_minibatch', 'mtl_ffnn_mini_batch', 'mtl_ffnn_hard_para_sharing', 'ELLA_ffnn_simple', 'ELLA_ffnn_tensorfactor', 'cnn_batch', 'cnn_minibatch'

# model_architecutre 'ffnn_batch'
#       model_hyperpara : 'hidden_layer'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ffnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'mtl_ffnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'mtl_ffnn_hard_para_sharing'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'task_specific_layer'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'mtl_ffnn_tensorfactor'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ELLA_ffnn_simple'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ELLA_ffnn_linear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ELLA_ffnn_linear_relation2'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'task_specific', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ELLA_ffnn_nonlinear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'ELLA_ffnn_nonlinear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'task_specific', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'

# model_architecutre 'cnn_batch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'cnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecutre 'mtl_cnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'


use_gpu, gpu_device_num = True, 0
mat_file_name = 'training_result1.mat'




data_type = 'mnist'
data_hyperpara = {}
data_hyperpara['num_train_data'] = 80
data_hyperpara['num_valid_data'] = 40
data_hyperpara['num_test_data'] = 1800
data_hyperpara['num_train_group'] = 25
#data_file_name = data_type + '_data_ind_split(' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + ').pkl'
#data_file_name = data_type + '_data_(task5_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + ').pkl'
data_file_name = 'mnist_new_data_testing.pkl'

### Generate/Load Data
num_train_max, num_valid_max, num_test_max = data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data']
if data_type is 'sine':
    train_data, validation_data, test_data = sine_data(data_file_name, data_hyperpara['num_task'], data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['param_for_data_generation'])
    sine_data_print_info(train_data, validation_data, test_data)
    classification_prob=False
elif data_type is 'sine_plus_linear':
    train_data, validation_data, test_data = sine_plus_linear_data(data_file_name, data_hyperpara['num_task'], data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['param_for_data_generation'])
    sine_data_print_info(train_data, validation_data, test_data)
    classification_prob=False
elif data_type is 'mnist':
    train_data, validation_data, test_data = mnist_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'])
    mnist_data_print_info(train_data, validation_data, test_data)
    classification_prob=True






train_hyperpara = {}
train_hyperpara['num_run_per_model'] = 13
train_hyperpara['lr'] = 0.01
train_hyperpara['lr_decay'] = 100.0
train_hyperpara['learning_step_max'] = 7500
#train_hyperpara['improvement_threshold'] = 0.9985    # for error (minimizing it)
train_hyperpara['improvement_threshold'] = 1.002      # for accuracy (maximizing it)
train_hyperpara['patience'] = 200
train_hyperpara['patience_multiplier'] = 3





#### Training one model
model_architecture = 'mtl_cnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16, 1]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 4, 4]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [32, 64]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num)


#### Training one model
model_architecture = 'mtl_ffnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 32]
model_hyperpara['batch_size'] = 20

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num)



#### Training one model
model_architecture = 'mtl_cnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16, 1]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 4, 4]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [32, 64]
model_hyperpara['padding_type'] = 'VALID'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)
