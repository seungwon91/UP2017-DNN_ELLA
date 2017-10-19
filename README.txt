
model_architecture = 'mtl_ffnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20




model_architecture = 'mtl_cnn_minibatch'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
model_hyperpara['stride_sizes'] = [2, 2, 2, 2]
model_hyperpara['channel_sizes'] = [32, 64]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]




model_architecture = 'ELLA_cnn_linear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 4, 4]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [32, 64]
model_hyperpara['cnn_KB_sizes'] = [9, 16, 9, 24]
model_hyperpara['fc_KB_sizes'] = [64, 64, 32, 64, 8, 16]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0, 0.0001]
## valid 0.54/test 0.52




model_architecture = 'ELLA_cnn_nonlinear_relation2'
model_hyperpara = {}
model_hyperpara['hidden_layer'] = [32, 16]
model_hyperpara['batch_size'] = 20
model_hyperpara['kernel_sizes'] = [5, 5, 4, 4]
model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
model_hyperpara['channel_sizes'] = [32, 64]
model_hyperpara['cnn_KB_sizes'] = [16, 24, 9, 32]
model_hyperpara['fc_KB_sizes'] = [128, 64, 64, 128, 8, 8]
model_hyperpara['padding_type'] = 'SAME'
model_hyperpara['max_pooling'] = True
model_hyperpara['pooling_size'] = [2, 2, 2, 2]
model_hyperpara['dropout'] = True
model_hyperpara['image_dimension'] = [28, 28, 1]
model_hyperpara['regularization_scale'] = [0.0001, 0.0001]
## valid 0.64/test 0.60




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
## valid 0.795/test 0.784





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
## valid 0.91/test 0.899 (lr 0.01 k KB 3,32,3,16 k TS 3,3 k st 2,2,2,2 fc 32,16 fc KB 64,64,32,64,8,16
## valid 0.795/test 0.768 (lr 0.05 k KB 3,32,3,16 k TS 3,3 k st 2,2,2,2 fc 32,16 fc KB 64,64,32,64,8,16
