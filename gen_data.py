import os
import pickle
from random import shuffle

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

############ Return Format ############
# (data_list, theta_list)
# data_list = [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)]
# theta_list = [theta_1, theta_2, ..., theta_t]
########## Return Format End ##########

#### function to print information of data file (number of parameters, dimension, etc.)
def sine_data_print_info(train_data, valid_data, test_data, print_info=True):
    assert (len(train_data)==len(valid_data) and len(train_data)==len(test_data)), "Different number of tasks in train/validation/test data"
    num_task = len(train_data)
    
    num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [validation_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
    x_dim, y_dim = train_data[0][0].shape[1], train_data[0][1].shape[1]
    if print_info:
        print "Tasks : ", num_task, "\nTrain data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test
        print "Input dim : ", x_dim, ", Output dim : ", y_dim, "\n"
    return (num_task, num_train, num_valid, num_test, x_dim, y_dim)


# y = a sin(x * theta) // MTL <- diff. theta
# data_param : [input_dim, a, x_min, x_max, theta_min, theta_max, noise amplitude]
def sine_data(data_file_name, num_task, num_train_max, num_valid_max, num_test_max, data_param=None, return_theta=False):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            train_data, validation_data, test_data, theta_for_tasks = pickle.load(fobj)
            print 'Successfully load data'
    else:
        train_data, validation_data, test_data = [], [], []
        theta_for_tasks = []

        for task_cnt in range(num_task):
            #### train data
            task_theta = (data_param[5]-data_param[4])*np.random.rand(data_param[0])+data_param[4]
            thate_for_tasks.append(task_theta)

            train_x = (data_param[3]-data_param[2])*np.random.rand(num_train_max, data_param[0])+data_param[2]
            noise = data_param[6]*np.random.randn(num_train_max)
            train_y = np.transpose([data_param[1] * np.sin(np.matmul(train_x, task_theta)) + noise])
            train_data.append( (train_x, train_y) )

            #### validation data
            valid_x = (data_param[3]-data_param[2])*np.random.rand(num_valid_max, data_param[0])+data_param[2]
            noise = data_param[6]*np.random.randn(num_valid_max)
            valid_y = np.transpose([data_param[1] * np.sin(np.matmul(valid_x, task_theta)) + noise])
            validation_data.append( (valid_x, valid_y) )

            #### test data
            test_x = (data_param[3]-data_param[2])*np.random.rand(num_test_max, data_param[0])+data_param[2]
            noise = data_param[6]*np.random.randn(num_test_max)
            test_y = np.transpose([data_param[1] * np.sin(np.matmul(test_x, task_theta)) + noise])
            test_data.append( (test_x, test_y) )

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data, theta_for_tasks], fobj)
            print 'Successfully generate/save data'

    sine_data_print_info(train_data, validation_data, test_data)
    if return_theta:
        return (train_data, validation_data, test_data, theta_for_tasks)
    else:
        return (train_data, validation_data, test_data)


# y = a sin(x*theta) + x*A (MTL <- diff. theta)
# data_param : [input_dim, a(scalar), A(numpy vec), x_min, x_max, theta_min, theta_max, noise amplitude]
def sine_plus_linear_data(data_file_name, num_task, num_train_max, num_valid_max, num_test_max, data_param=None, return_theta=False):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            train_data, validation_data, test_data, theta_for_tasks = pickle.load(fobj)
            print 'Successfully load data'
    else:
        train_data, validation_data, test_data = [], [], []
        theta_for_tasks = []

        for task_cnt in range(num_task):
            #### train data
            task_theta = (data_param[6]-data_param[5])*np.random.rand(data_param[0])+data_param[5]
            thate_for_tasks.append(task_theta)

            train_x = (data_param[4]-data_param[3])*np.random.rand(num_train_max, data_param[0])+data_param[3]
            noise = data_param[7]*np.random.randn(num_train_max)
            train_y = np.transpose([data_param[1] * np.sin(np.matmul(train_x, task_theta)) + np.matmul(train_x, data_param[2]) + noise])
            train_data.append( (train_x, train_y) )

            #### validation data
            valid_x = (data_param[4]-data_param[3])*np.random.rand(num_valid_max, data_param[0])+data_param[3]
            noise = data_param[7]*np.random.randn(num_valid_max)
            valid_y = np.transpose([data_param[1] * np.sin(np.matmul(valid_x, task_theta)) + np.matmul(valid_x, data_param[2]) + noise])
            validation_data.append( (valid_x, valid_y) )

            #### test data
            test_x = (data_param[4]-data_param[3])*np.random.rand(num_test_max, data_param[0])+data_param[3]
            noise = data_param[7]*np.random.randn(num_test_max)
            test_y = np.transpose([data_param[1] * np.sin(np.matmul(test_x, task_theta)) + np.matmul(test_x, data_param[2]) + noise])
            test_data.append( (test_x, test_y) )

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data, theta_for_tasks], fobj)
            print 'Successfully generate/save data'

    sine_data_print_info(train_data, validation_data, test_data)
    if return_theta:
        return (train_data, validation_data, test_data, theta_for_tasks)
    else:
        return (train_data, validation_data, test_data)

# MNIST data (label : number of train/number of valid/number of test)
# 0 : 5444/479/980, 1 : 6179/563/1135, 2 : 5470/488/1032, 3 : 5638/493/1010
# 4 : 5307/535/982, 5 : 4987/434/892,  6 : 5417/501/958,  7 : 5715/550/1028
# 8 : 5389/462/974, 9 : 5454/495/1009

#### function to split data into each categories (gather data of same digit)
def mnist_data_class_split(mnist_class):
    train_img, valid_img, test_img = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    for cnt in range(mnist_class.train.images.shape[0]):
        class_inst, x = mnist_class.train.labels[cnt], mnist_class.train.images[cnt, :]
        train_img[class_inst].append(x)

    for cnt in range(mnist_class.validation.images.shape[0]):
        class_inst, x = mnist_class.validation.labels[cnt], mnist_class.validation.images[cnt, :]
        valid_img[class_inst].append(x)

    for cnt in range(mnist_class.test.images.shape[0]):
        class_inst, x = mnist_class.test.labels[cnt], mnist_class.test.images[cnt, :]
        test_img[class_inst].append(x)
    return (train_img, valid_img, test_img)

#### function to make dataset (either train/valid/test) for binary classification
def mnist_data_gen_binary_classification(img_for_true, img_for_false, dataset_size):
    #### dataset has at least 'min_num_from_each' numbers of instances from each class
    #### thus, the number of data for a class is [min_num_from_each, min_num_from_each + num_variable_class]
    min_num_from_each = int(0.4*dataset_size)
    num_variable_class = dataset_size - 2*min_num_from_each
    if (min_num_from_each+num_variable_class < len(img_for_true)) and (min_num_from_each+num_variable_class < len(img_for_false)):
        num_true = min_num_from_each + np.random.randint(0, num_variable_class+1, size=1)[0]
    elif (min_num_from_each+num_variable_class < len(img_for_true)):
        tmp = np.random.randin(0, len(img_for_false)+1-min_num_from_each, size=1)[0]
        num_true = min_num_from_each + num_variable_class - tmp
    elif (min_num_from_each+num_variable_class < len(img_for_false)):
        num_true = min_num_from_each + np.random.randint(0, len(img_for_true)+1-min_num_from_each, size=1)[0]
    else:
        tmp = np.random.randint(num_variable_class+min_num_from_each-len(img_for_false), len(img_for_true)-min_num_from_each, size=1)[0]
        num_true = min_num_from_each + tmp

    indices_for_true, indices_for_false = range(len(img_for_true)), range(len(img_for_false))
    shuffle(indices_for_true)
    shuffle(indices_for_false)

    indices_classes = [1 for _ in range(num_true)] + [0 for _ in range(dataset_size-num_true)]
    shuffle(indices_classes)

    data_x, data_y, cnt_false = [], [], 0
    for cnt in range(dataset_size):
        if indices_classes[cnt] == 0:
            data_x.append(img_for_false[indices_for_false[cnt_false]])
            data_y.append(0)
            cnt_false = cnt_false+1
        else:
            data_x.append(img_for_true[indices_for_true[cnt-cnt_false]])
            data_y.append(1)
    return (data_x, data_y)

#### function to print information of data file (number of parameters, dimension, etc.)
def mnist_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], train_data[0][1].shape[1]
        if print_info:
            print "Tasks : ", num_task, "\nTrain data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test
            print "Input dim : ", x_dim, ", Output dim : ", y_dim, "\n"
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim)
    else:
        assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], train_data[0][0][1].shape[1]
        if print_info:
            print "Tasks : ", num_task, ", Groups of training/valid : ", num_group, "\nTrain data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test
            print "Input dim : ", x_dim, ", Output dim : ", y_dim, "\n"
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim)


#### generate/handle data of mnist
#### data format (train_data, validation_data, test_data)
####    - train/validation : [group1(list), group2(list), ... ] with the group of test data format
####    - test : [(task1_x, task1_y), (task2_x, task2_y), ... ]
def mnist_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            train_data, validation_data, test_data = pickle.load(fobj)
            print 'Successfully load data'
    else:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        #### subclasses : train, validation, test with images/labels subclasses
        #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
        num_tasks = 5
        categorized_train_x, categorized_valid_x, categorized_test_x = mnist_data_class_split(mnist)

        ## process train data
        train_data = []
        for group_cnt in range(num_train_group):
            train_data_tmp = []
            for task_cnt in range(num_tasks):
                train_x_tmp, train_y_tmp = mnist_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                train_data_tmp.append( ( np.array(train_x_tmp), np.reshape(np.array(train_y_tmp), (len(train_y_tmp), 1)) ) )
            train_data.append(train_data_tmp)

        ## process validation data
        validation_data = []
        for group_cnt in range(num_train_group):
            validation_data_tmp = []
            for task_cnt in range(num_tasks):
                valid_x_tmp, valid_y_tmp = mnist_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                validation_data_tmp.append( ( np.array(valid_x_tmp), np.reshape(np.array(valid_y_tmp), (len(valid_y_tmp), 1)) ) )
            validation_data.append(validation_data_tmp)

        ## process test data
        test_data = []
        for task_cnt in range(num_tasks):
            test_x_tmp, test_y_tmp = mnist_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
            test_data.append( ( np.array(test_x_tmp), np.reshape(np.array(test_y_tmp), (len(test_y_tmp), 1)) ) )

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print 'Successfully generate/save data'

    mnist_data_print_info(train_data, validation_data, test_data)
    return (train_data, validation_data, test_data)
