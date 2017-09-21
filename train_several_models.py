from os import listdir, getcwd, mkdir
from math import sqrt

from numpy import inf
from tensorflow import reset_default_graph

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
    

def print_to_txt(txt_file_name, model_architecture, model_hyperpara, train_time, best_epoch, best_valid_error, best_test_error):
    if model_architecture is 'ffnn_batch':
        model_summary = "Model : %s\nHidden : %s\n" %(model_architecture, str(model_hyperpara['hidden_layer']))
    elif (model_architecture is 'ffnn_minibatch') or (model_architecture is 'mtl_ffnn_mini_batch'):
        model_summary = "Model : %s\nHidden : %s\n" %(model_architecture, str(model_hyperpara['hidden_layer']))
    elif (model_architecture is 'mtl_ffnn_hard_para_sharing'):
        model_summary = "Model : %s\nHidden : %s\tTask Specific : %s\n" %(model_architecture, str(model_hyperpara['hidden_layer']), str(model_hyperpara['task_specific_layer']))
    elif (model_architecture is 'ELLA_ffnn_simple') or (model_architecture is 'mtl_ffnn_tensorfactor') or (model_architecture is 'ELLA_ffnn_linear_relation') or (model_architecture is 'ELLA_ffnn_nonlinear_relation'):
        model_summary = "Model : %s\nHidden : %s\tKB : %s\tReg Scale : %s\n" %(model_architecture, str(model_hyperpara['hidden_layer']), str(model_hyperpara['knowledge_base']), str(model_hyperpara['regularization_scale']))
    elif (model_architecture is 'ELLA_FFNN_linear_relation2') or (model_architecture is 'ELLA_ffnn_nonlinear_relation2'):
        model_summary = "Model : %s\nHidden : %s\tKB : %s\tTS : %s\tReg Scale : %s\n" %(model_architecture, str(model_hyperpara['hidden_layer']), str(model_hyperpara['knowledge_base']), str(model_hyperpara['task_specific']), str(model_hyperpara['regularization_scale']))

    #result_summary = "Time consumption for training : %.2f\nBest validation/test error : %.4f/ %.4f (at epoch %d)\n\n" %(train_time, best_valid_error, best_test_error, best_epoch)
    #result_summary = "Time consumption for training : %.2f (%.3f)\nBest validation/test error : %.4f (%.4f)/ %.4f (%.4f)\n\n" %(mean_of_list(train_time), stddev_of_list(train_time), mean_of_list(best_valid_error), stddev_of_list(best_valid_error), mean_of_list(best_test_error), stddev_of_list(best_test_error))
    result_summary = "Time consumption for training : %.2f (%.3f)\nBest validation/test error : %.4f / %.4f\nMean validation/test error : %.4f (%.4f)/ %.4f (%.4f)\n\n" %(mean_of_list(train_time), stddev_of_list(train_time), max(best_valid_error), max(best_test_error), mean_of_list(best_valid_error), stddev_of_list(best_valid_error), mean_of_list(best_test_error), stddev_of_list(best_test_error))

    if not ('txt_result' in listdir(getcwd())):
        mkdir('./txt_result')

    with open('./txt_result/' + txt_file_name, 'a') as fobj:
        fobj.write(model_summary+result_summary)


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

num_run = 11

#result_txt_name = 'comparison_of_relation_btw_KB_and_TS(trD80).txt'
result_txt_name = 'delete_this1.txt'
data_type = 'mnist'
data_hyperpara = {}
data_hyperpara['num_train_data'] = 80
data_hyperpara['num_valid_data'] = 40
data_hyperpara['num_test_data'] = 1800
#data_file_name = data_type + '_data_ind_split(' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + ').pkl'
#data_file_name = data_type + '_data_(task5_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + ').pkl'
data_file_name = 'mnist_new_data_testing.pkl'


save_result = False
train_hyperpara = {}
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
'''
model_architecture = 'mtl_ffnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 32]
model_hyperpara['batch_size'] = 20
'''
train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
reset_default_graph()





'''
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "1-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "1-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [8, 4, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "1-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "1-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [32, 16, 8]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "1-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)






#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "2-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "2-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [8, 4, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "2-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "2-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [32, 16, 8]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "2-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)







#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "3-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "3-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [8, 4, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "3-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "3-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [32, 16, 8]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "3-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)







#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "4-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "4-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [8, 4, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "4-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [16, 8, 4]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "4-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [16, 8, 4]
model_hyperpara['task_specific'] = [32, 16, 8]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "4-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################





#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "5-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "5-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [6, 3, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "5-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "5-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "5-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)









#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "6-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "6-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [6, 3, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "6-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "6-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "6-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)







#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "7-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "7-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [6, 3, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "7-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "7-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "7-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)





#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "8-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "8-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [6, 3, 2]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "8-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "8-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [24, 12]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [12, 6, 3]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "8-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################





#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "9-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "9-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "9-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "9-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [48, 24, 12]
model_hyperpara['regularization_scale'] = [0.0, 0.001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "9-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)









#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "10-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "10-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "10-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "10-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [48, 24, 12]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "10-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)







#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "11-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "11-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "11-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "11-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [48, 24, 12]
model_hyperpara['regularization_scale'] = [0.0, 0.00001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "11-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)





#### Training one model
model_architecture = 'ELLA_ffnn_linear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "12-1-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "12-2-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [12, 6, 3]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "12-3-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [24, 12, 6]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "12-4-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)


#### Training one model
model_architecture = 'ELLA_ffnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [48, 24]
model_hyperpara['batch_size'] = 20
model_hyperpara['knowledge_base'] = [24, 12, 6]
model_hyperpara['task_specific'] = [48, 24, 12]
model_hyperpara['regularization_scale'] = [0.0, 0.000001]

#best_one = 0.0
train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = [], [], [], []
for cnt in range(num_run):
    train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp = train_main(data_type=data_type, data_file_name=data_file_name, data_hyperpara=data_hyperpara, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_hyperpara=train_hyperpara, save_result=save_result, useGPU=use_gpu, GPU_device=gpu_device_num)
    reset_default_graph()

    train_time.append(train_time_tmp)
    best_epoch.append(best_epoch_tmp)
    best_valid_accuracy.append(best_valid_accuracy_tmp)
    test_accuracy_at_best_epoch.append(test_accuracy_at_best_epoch_tmp)
    #if best_one < test_accuracy_at_best_epoch_tmp:
    #    train_time, best_epoch, best_valid_accuracy, test_accuracy_at_best_epoch = train_time_tmp, best_epoch_tmp, best_valid_accuracy_tmp, test_accuracy_at_best_epoch_tmp
    #    best_one = test_accuracy_at_best_epoch_tmp
    print "12-5-%d\n\n" % (cnt)

print_to_txt(txt_file_name=result_txt_name, model_architecture=model_architecture, model_hyperpara=model_hyperpara, train_time=train_time, best_epoch=best_epoch, best_valid_error=best_valid_accuracy, best_test_error=test_accuracy_at_best_epoch)
'''
