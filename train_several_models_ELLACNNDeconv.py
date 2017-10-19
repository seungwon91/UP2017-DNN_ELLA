from os import listdir, getcwd, mkdir
from os.path import isfile

from scipy.io import loadmat, savemat
import numpy as np
from tensorflow import reset_default_graph

from gen_data import sine_data, sine_plus_linear_data, sine_data_print_info, mnist_data, mnist_data_print_info
from train_main import train_main
from utils import reformat_result_for_mat


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
        train_result_tmp = train_main(model_architecture, model_hyperpara, train_hyperpara, [dataset[0][group_cnt[run_cnt]], dataset[1][group_cnt[run_cnt]], dataset[2]], data_type, classification_prob, useGPU, GPU_device)
        result_from_train_run.append(train_result_tmp)
        print("%d-th training run\n\n" % (run_cnt+1))
        reset_default_graph()

    #### save training summary
    result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, group_cnt)
    saved_result[-1] = result_of_curr_run
    savemat('./Result/'+mat_file_name, {'training_summary':saved_result})

    return saved_result



# input arguments : data_type, data_file_name, data_hyperpara, model_architecture, model_hyperpara, train_hyperpara, save_result, result_folder_name=None

# model_architecture : 'ffnn_batch', 'ffnn_minibatch', 'mtl_ffnn_mini_batch', 'mtl_ffnn_hard_para_sharing', 'ELLA_ffnn_simple', 'ELLA_ffnn_tensorfactor', 'cnn_batch', 'cnn_minibatch'

# model_architecture 'ffnn_batch'
#       model_hyperpara : 'hidden_layer'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ffnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'mtl_ffnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'mtl_ffnn_hard_para_sharing'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'task_specific_layer'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'mtl_ffnn_tensorfactor'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_ffnn_simple'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_ffnn_linear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_ffnn_linear_relation2'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'task_specific', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_ffnn_nonlinear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_ffnn_nonlinear_relation'
#       model_hyperpara : 'hidden_layer', 'batch_size', 'knowledge_base', 'task_specific', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'

# model_architecture 'cnn_batch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'cnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'mtl_cnn_minibatch'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_cnn_linear_relation2' or 'ELLA_cnn_nonlinear_relation2'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'cnn_KB_sizes', 'fc_KB_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'
# model_architecture 'ELLA_cnn_deconv_relation_relu' or 'ELLA_cnn_deconv_relation_tanh'
#       model_hyperpara : 'hidden_layer', 'kernel_sizes', 'stride_sizes', 'channel_sizes', 'cnn_KB_sizes', 'cnn_TS_sizes', 'cnn_deconv_stride_sizes', 'fc_KB_sizes', 'padding_type', 'max_pooling', 'pooling_size', 'dropout', 'image_dimension', 'batch_size', 'regularization_scale'
#       train_hyperpara : 'lr', 'learning_step_max', 'improvement_threshold', 'patience', 'patience_multiplier'


use_gpu, gpu_device_num = True, 0
mat_file_name = 'mnist_mtl_ellacnndeconv.mat'


##############################################
### Generate/Load Data
##############################################
data_type = 'mnist'
data_hyperpara = {}
data_hyperpara['num_train_data'] = 80
data_hyperpara['num_valid_data'] = 40
data_hyperpara['num_test_data'] = 1800
data_hyperpara['num_train_group'] = 31
#data_hyperpara['one_hot_encoding'] = True
data_file_name = 'mnist_mtl_data_group_1Dlabel.pkl'


#data_hyperpara['num_train_data'] = 200
#data_hyperpara['num_valid_data'] = 80
#data_file_name = 'mnist_mtl_data_group(200_80_1800).pkl'


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




##############################################
### Training/Model hyperparameter set-up
##############################################

train_hyperpara = {}
train_hyperpara['num_run_per_model'] = 5
train_hyperpara['lr'] = 0.025
train_hyperpara['lr_decay'] = 200.0
train_hyperpara['learning_step_max'] = 4000
#train_hyperpara['improvement_threshold'] = 0.9985    # for error (minimizing it)
train_hyperpara['improvement_threshold'] = 1.002      # for accuracy (maximizing it)
train_hyperpara['patience'] = 200
train_hyperpara['patience_multiplier'] = 2.5


#train_hyperpara['num_run_per_model'] = 2
#train_hyperpara['learning_step_max'] = 10
#train_hyperpara['patience'] = 4





#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 64, 32, 64, 8, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num)


#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [64, 32]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 96, 48, 64, 16, 24]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [16, 8]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [32, 64, 24, 32, 8, 8]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



############ change CNN

#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 64, 32, 64, 8, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)


#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [64, 32]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 96, 48, 64, 16, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_relu'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [16, 8]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [32, 64, 24, 32, 8, 8]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)





######################################################################
################################### Switched to Tanh
######################################################################

#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 64, 32, 64, 8, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)


#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [64, 32]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 96, 48, 64, 16, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [16, 8]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [64, 32]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [32, 64, 24, 32, 8, 8]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



############ change CNN

#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 64, 32, 64, 8, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)


#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [64, 32]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [64, 96, 48, 64, 16, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)



#### Training one model
model_architecture = 'ELLA_cnn_deconv_relation_tanh'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [16, 8]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [48, 24]
model_hyperpara['cnn_KB_sizes'] = [3, 32, 3, 16]
model_hyperpara['cnn_TS_sizes'] = [3, 3]
model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['fc_KB_sizes'] = [32, 64, 24, 32, 8, 8]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 1e-7]

saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, data_hyperpara['num_train_group'], mat_file_name, classification_prob, saved_result=saved_result, useGPU=use_gpu, GPU_device=gpu_device_num)

