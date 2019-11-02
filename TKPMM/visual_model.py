import numpy as np
import json
import os
import cv2
import tensorflow as tf
from tensorflow import pywrap_tensorflow
import re
import random
import tensorflow.contrib.slim as slim

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Config():
    _BATCH_NORM_DECAY = 0.0
    _BATCH_NORM_EPSILON = 1e-5
    data_format = 'channels_last'
    training = True
    pre_activation = True
    block_sizes = [3, 4, 6, 3]
    block_strides = [1, 2, 2, 2]
    num_filters = 64
    kernel_size = 3
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    hot = 1.0
    batch_size = 50
    lr = 0.0007
    image_height = 256
    image_width = 128
    weight_decay = 0.0005
    init_scale = 0.04
    max_grad_norm = 15
    momentum = 0.9   
    epoches = 30001
    data_dir = ''   # training data dir
    test_dir = ''   #  test data dir
    output_dir = ''  # model saving dir

config = Config()

class pixel_attention_model():
    def batch_norm(self, inputs, training, data_format):
        return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
            scale=True, training=training, fused=True)

    def fixed_padding(self, inputs, kernel_size, data_format):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)
        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def _building_block_v1(self, inputs, filters, training, projection_shortcut, strides,
                           data_format):
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = self.batch_norm(inputs=shortcut, training=training,
                                  data_format=data_format)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = inputs + shortcut
        inputs = tf.nn.relu(inputs)
        return inputs

    def block_layer(self, inputs, filters, blocks, strides,
                       training, name, data_format):
        filters_out = filters
        def projection_shortcut(inputs):
            return self.conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                data_format=data_format)
        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = self.block_fn(inputs, filters, training, projection_shortcut, strides,
                          data_format)
        for _ in range(1, blocks):
            inputs = self.block_fn(inputs, filters, training, None, 1, data_format)
        layer_shape = inputs.get_shape().as_list()
        return tf.identity(inputs, name), layer_shape

    def difference_map(self, feature_map):
        feature_y = tf.transpose(feature_map[1::2, :, :, :], (0, 3, 1, 2))
        y_shape = feature_y.get_shape().as_list()
        feature_x = feature_map[::2, :, :, :]
        x_shape = feature_x.get_shape().as_list()
        x_temp = tf.reshape(feature_x, [x_shape[0], x_shape[1] * x_shape[2], x_shape[3]])  # [batch, 16*8, 256]
        y_temp = tf.reshape(feature_y, [y_shape[0], y_shape[1], y_shape[2] * y_shape[3]])  # [batch, 256, 16*8]
        kronecker_temp = tf.matmul(x_temp, y_temp)
        kronecker_shape = kronecker_temp.get_shape().as_list()
        kronecker_map_soft = tf.transpose(kronecker_temp, [0, 2, 1])
        diff_map = feature_x - tf.reshape(tf.transpose(tf.matmul(y_temp, kronecker_map_soft), [0, 2, 1]),
                                          [x_shape[0], x_shape[1], x_shape[2], -1])
        return diff_map

    def get_kernel_size(self, factor):
        """
        Find the kernel size given the desired factor of upsampling
        """
        return 2 * factor - factor % 2

    def get_upsample_filter(self, filter_shape, upscale_factor):
        """
        Make a 2D bilinear kernel
        """
        ### filter_shape is [ width, height, num_in_channel, num_out_channel ]
        kernel_size = filter_shape[1]
        ### center location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            center_location = upscale_factor - 1
        else:
            center_location = upscale_factor - 0.5  # e.g..
        bilinear_grid = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ### Interpolation Calculation
                value = (1 - abs((x - center_location) / upscale_factor)) * (
                    1 - abs((y - center_location) / upscale_factor))
                bilinear_grid[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear_grid
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        bilinear_weights = tf.get_variable(name="deconv_bilinear_filter", initializer=init, shape=weights.shape)
        return bilinear_weights

    def upsample_layer(self, bottom_map, name, upscale_factor, shape):
        """
        The spatial extent of the output map can be optained from the fact that (upscale_factor -1) pixles are inserted between two successive pixels
        """
        kernel_size = self.get_kernel_size(upscale_factor)
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        # data tensor: 4D tensors are usually: [BATCH, Height, Width, Channel]
        n_channels = list(bottom_map.get_shape())[-1]
        with tf.variable_scope(name) as scope:
            # shape of the bottom tensor
            if shape is None:
                in_shape = tf.shape(bottom_map)
                print("in_shape", in_shape.get_shape().as_list)
                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, n_channels]
            else:
                new_shape = [shape[0], shape[1], shape[2], shape[3]]
            output_shape = tf.stack(new_shape)
            filter_shape = [kernel_size, kernel_size, new_shape[3], n_channels]
            weights_ = self.get_upsample_filter(filter_shape, upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom_map, weights_, output_shape, strides=strides, padding='SAME')
            return deconv

    def attention_model(self, feature_map):  
        feature_map_shape = list(feature_map.get_shape())  # [batch_size, H, W, n_channels]
        target_shape = self.resnetlayer_shape[list(self.resnetlayer_shape.keys())[-1]]  # [16, 8, 512]
        strides = int(int(feature_map_shape[1]) / target_shape[1])
        kernel_size = int(strides * 2)
        pad_0 = int(strides / 2)
        feature_map_pad = tf.pad(feature_map, [[0, 0], [pad_0, pad_0], [pad_0, pad_0], [0, 0]])
        feature_conv = tf.layers.conv2d(
            inputs=feature_map_pad, filters=1, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=self.data_format)
        feature_conv = self.batch_norm(feature_conv, self.training, self.data_format)
        feature_conv = tf.nn.relu(feature_conv)
        feature_map_shape[-1] = 1
        upscore_name = "upattention2_" + str(feature_map_shape[1])
        feature_deconv = self.upsample_layer(feature_conv, upscore_name, strides, feature_map_shape)
        feature_deconv = self.batch_norm(feature_deconv, self.training, self.data_format)
        feature_deconv = tf.nn.relu(feature_deconv)
        feature_deconv_expend = tf.reshape(
            tf.tile(feature_deconv, multiples=[1, 1, 1, list(feature_map.get_shape())[-1]]),
            [feature_map_shape[0], feature_map_shape[1] * feature_map_shape[2], -1])
        attention = tf.reshape(tf.nn.softmax(feature_deconv_expend / self.hot, dim=1),
                               shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], -1])
        return attention  # shape:[batch_size, H, W, n_channels] need not to be expended

    def after_resnet_upsample(self, input, resnet_layer_index):
        upsample_shape = self.resnetlayer_shape[resnet_layer_index - 1]       # last layer:3
        upscore_name = "upscore2_" + str(resnet_layer_index)
        feature_deconv = self.upsample_layer(bottom_map=input, name=upscore_name, upscale_factor=2, shape=upsample_shape)
        feature_deconv = self.batch_norm(feature_deconv, self.training, self.data_format)
        feature_deconv = tf.nn.relu(feature_deconv)
        filling_map = tf.layers.conv2d(
            inputs=self.resnet_feature[resnet_layer_index - 1], filters=upsample_shape[-1], kernel_size=1, strides=1,
            padding='SAME', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=self.data_format)
        upsample_map = tf.add(feature_deconv, filling_map)
        return upsample_map

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.weight_decay = config.weight_decay
        self._BATCH_NORM_DECAY = config._BATCH_NORM_DECAY
        self._BATCH_NORM_EPSILON = config._BATCH_NORM_EPSILON
        self.images = tf.placeholder(tf.float32, [self.batch_size * 2, self.image_height, self.image_width, 3],
                                     name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 2], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=None)
        self.block_fn = self._building_block_v1
        self.data_format = config.data_format
        self.num_filters = config.num_filters   # The number of filters to use for the first block layer
        self.kernel_size = config.kernel_size   # kernel_size: The kernel size to use for convolution.
        self.conv_stride = config.conv_stride
        self.block_sizes = config.block_sizes
        self.act_margin = 0.05
        self.pre_activation = config.pre_activation
        self.block_strides = config.block_strides   # List of integers representing the desired stride size for
        self.training = tf.placeholder(tf.bool)
        self.first_pool_size = config.first_pool_size
        self.first_pool_stride = config.first_pool_stride
        self.hot = config.hot
        self.dtype = tf.float32
        self.momentum = config.momentum
        self.resnetlayer_shape = {}   # image shape after each block
        self.resnet_feature = {}    # the output of the 0,1,2,3,4-th layer
        self.upsample_feature = {}
        self.before_global = {}
        self.dtype = tf.float32
        images = tf.identity(self.images, 'origin_inputs')
        inputs = tf.cast(images, dtype=self.dtype)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
        # inputs = tf.identity(inputs, 'initial_conv')
        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first ResNet unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection. Cf. Appendix of [2].
        inputs = self.batch_norm(inputs, self.training, self.data_format)
        inputs = tf.nn.relu(inputs)
        self.resnet_feature[0] = inputs
        self.resnetlayer_shape[0] = inputs.get_shape().as_list()
        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                inputs=inputs, pool_size=self.first_pool_size,
                strides=self.first_pool_stride, padding='SAME',
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2 ** i)
            inputs, self.resnetlayer_shape[i + 1] = self.block_layer(
                inputs=inputs, filters=num_filters, blocks=num_blocks,
                strides=self.block_strides[i], training=self.training,
                name='block_layer{}'.format(i + 1), data_format=self.data_format)
            self.resnet_feature[i + 1] = inputs
        diff_map_pre = self.difference_map(feature_map=inputs)
        
        diff_map = diff_map_pre
        self.before_global[-1] = diff_map
		
        for i in range(len(self.block_sizes) - 1):
            inputs = self.after_resnet_upsample(inputs, resnet_layer_index=len(self.block_sizes) - i)
            self.upsample_feature[i] = inputs
            feature_x_index = [i for i in range(self.batch_size * 2) if i % 2 == 0]
            # print(len(feature_x_index))
            if len(feature_x_index) == 1:
                feature_x = inputs[feature_x_index[0], :, :, :]
                feature_x_shape = list(feature_x.get_shape())
                feature_x = tf.reshape(feature_x, [1, feature_x_shape[0], feature_x_shape[1], feature_x_shape[2]])
            else:
                feature_x = inputs[::2, :, :, :]
            attention_layer_map = self.attention_model(feature_map=feature_x)
            diff_map_pre = self.difference_map(feature_map=inputs)
            if self.training == False:
                diff_map_pre_shape = list(diff_map_pre[0].get_shape())
                diff_map = tf.reshape(diff_map_pre[0],
                                      [1, diff_map_pre_shape[0], diff_map_pre_shape[1], diff_map_pre_shape[2]])
            else:
                diff_map = diff_map_pre
            diff_attention_layer_map = tf.multiply(diff_map, attention_layer_map)
            self.before_global[i] = diff_attention_layer_map
			
        self.temp = {}
        for i in range(-1, len(self.block_sizes) - 1):
            tmp = tf.reduce_mean(self.before_global[i], [1, 2])  # [batch_size, 512]
            tmp_shape = list(tmp.get_shape())
            tmp_double = tf.multiply(tmp, tmp)
            self.temp[i] = tf.layers.batch_normalization( 
                inputs=tmp_double, axis=1,
                momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
                scale=True, training=self.training, fused=True)

        self.last_map = self.temp[-1]
        for i in range(0, len(self.block_sizes) - 1):
            self.last_map = tf.concat([self.last_map, self.temp[i]], 1)  # [batch_size, 512 + 256 + 64 + 32 + 16 + 8]
        fc1 = tf.layers.dense(self.last_map, 50, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')
        self.logits = fc2
		# hard-mining : make a mask:
        self.cross_pre = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.cross_filter = tf.cast(self.cross_pre > self.act_margin, self.dtype)
        self.cross_entropy = tf.reduce_mean(tf.multiply(self.cross_filter, self.cross_pre))
        #self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        tf.identity(self.cross_entropy, name='cross_entropy')
        # precision&recall&accuracy
        precision_pre = tf.nn.softmax(self.logits)
        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(precision_pre,1), tf.argmax(self.labels,1)), tf.float32))
        # precision
        self.precision = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(precision_pre, 1), tf.zeros([self.batch_size], tf.int64)), tf.equal(tf.argmax(self.labels, 1), tf.zeros([self.batch_size], tf.int64))), tf.int64)) / tf.reduce_sum(tf.cast(tf.equal(tf.argmax(precision_pre, 1), tf.zeros([self.batch_size], tf.int64)), tf.int64))
        self.recall = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(precision_pre, 1), tf.zeros([self.batch_size], tf.int64)), tf.equal(tf.argmax(self.labels, 1), tf.zeros([self.batch_size], tf.int64))), tf.int64)) / tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.zeros([self.batch_size], tf.int64)), tf.int64))

        def exclude_batch_norm(name):
            return 'batch_normalization' not in name

        loss_filter_fn = None or exclude_batch_norm
        self.l2_loss = self.weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
             if loss_filter_fn(v.name)])
        self.loss = self.cross_entropy + self.l2_loss
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.lr,
            momentum=self.momentum)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train = optimizer.apply_gradients(zip(grads, tvars))


import time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def get_pair(path, num_id, positive, tongji):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break
    for i in range(2):
        filepath = ''
        while True:
            index = os.listdir(path + '/' + list(tongji.keys())[id[i]])[int(random.random() * tongji[list(tongji.keys())[id[i]]])]
            filepath = '%s/%s/%s' % (path, list(tongji.keys())[id[i]], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair

def get_num_id(tongji): # total number
    return len(tongji) - 1

def tong_ji(path):
    file_list = os.listdir(path)
    tongji = {}
    for i in file_list:
        tongji[i] = len(os.listdir(path + '/' + i))
    return tongji

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def extract_from_anno(p, img):
    anno_path = '/home/zpl/dulikongjian/labels_text_images/all_annotation/'
    xml_path = anno_path + p.split('/')[-2] + '/' + p.split('/')[-1].split('.')[0] + '.xml'
    #print(xml_path)
    #print(p)
    file = open(xml_path, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        if 'ymin' in line:
            ymin = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'ymax' in line:
            ymax = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'xmin' in line:
            xmin = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'xmax' in line:
            xmax = max(int(float(line.split('>')[1].split('<')[0])), 0)
        else:
            continue
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped

def read_data(path, num_id, tongji, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in range(batch_size):
        images = []
        sam = random.choice([1,2,3,4,5])
        if sam <= 2:
            pairs = get_pair(path, num_id, True, tongji)
            labels.append([1., 0.])
        else:
            pairs = get_pair(path, num_id, False, tongji)
            labels.append([0., 1.])
        for p in pairs:
            img = cv_imread(p)
            flip = random.randint(1,2)
            if flip == 2:  # data expension
                img = cv2.flip(img, 1)
            image = extract_from_anno(p, img)
            image = cv2.resize(image, (image_width, image_height))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        batch_images.append(images)
    return np.transpose(np.array(batch_images), (1, 0, 2, 3, 4)), np.array(labels)
######################################################################

def read_data_test(path, num_id, tongji, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    target = int(random.random() * num_id)
    while True:
        target_index = os.listdir(path + '/' + list(tongji.keys())[target])[int(random.random() * tongji[list(tongji.keys())[target]])]
        target_path = '%s/%s/%s' % (path, list(tongji.keys())[target], target_index)
        if not os.path.exists(target_path):
            continue
        break
    for i in range(batch_size):
        a = random.randint(1, 100)
        if a <= 20:
            positive = True
            labels.append([1., 0.])
        else:
            positive = False
            labels.append([0., 1.])
        pairs = get_pair_test(target, target_path, path, num_id, positive, tongji)
        images = []
        for p in pairs:
            img = cv_imread(p)
            image = extract_from_anno(p, img)
            image = cv2.resize(image, (image_width, image_height))
            images.append(image)
        batch_images.append(images)
    return np.transpose(np.array(batch_images), (1, 0, 2, 3, 4)), np.array(labels)

def get_pair_test(target, target_index, path, num_id, positive, tongji):
    pair = []
    pair.append(target_index)
    if positive:
        value = target
        id = [value, value]
    else:
        while True:
            id = int(random.random() * num_id)
            if id != target:
                id = [target, id]
                break
    filepath = ''
    while True:
        index = os.listdir(path + '/' + list(tongji.keys())[id[1]])[int(random.random() * tongji[list(tongji.keys())[id[1]]])]
        filepath = '%s/%s/%s' % (path, list(tongji.keys())[id[1]], index)
        if not os.path.exists(filepath):
            continue
        break
    pair.append(filepath)
    return pair
######################################################################
path = config.data_dir
set = 'train'
path_test = config.test_dir

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    if config.mode == 'train':
        #config.batch_size = 1
        initializer = tf.random_uniform_initializer(-Config.init_scale,
                                                    Config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mod = pixel_attention_model(config=config)
        tf.global_variables_initializer().run()
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars = bn_moving_vars + [g for g in g_list if 'moving_variance' in g.name]
        var_list = var_list + bn_moving_vars
        model_saver = tf.train.Saver(var_list=var_list, max_to_keep=50)
        # print('load model')
		# if you want to load a pre-trained model:
        # model_saver.restore(session, config.load_dir + '')
        # print('done')
        tongji = tong_ji(path)
        train_num_id = get_num_id(tongji)
        tongji_test = tong_ji(path_test)
        train_num_id_test = get_num_id(tongji_test)
        loss_curve = []
        for i in range(config.epoches):
            print('load data')
            batch_images, batch_labels = read_data(path=config.data_dir, num_id=train_num_id, tongji=tongji,
                                                   image_width=mod.image_width, image_height=mod.image_height,
                                                   batch_size=mod.batch_size)
            print(batch_images.shape)
            batch_data = np.zeros([batch_images.shape[1] * 2, batch_images.shape[2], batch_images.shape[3], batch_images.shape[4]])
            batch_data[::2, :, :, :] = batch_images[0]
            batch_data[1::2, :, :, :] = batch_images[1]
            print(batch_data.shape)
            #batch_data = batch_images
            print("Training Epoch: %d ..." % (i + 1))
            lr = config.lr * ((0.0001 * i + 1) ** -0.75)
            feed_dict = {mod.lr: lr, mod.images: batch_data,
                         mod.labels: batch_labels, mod.training: True}
            session.run(mod.train, feed_dict=feed_dict)
            session.run(mod.update_ops, feed_dict=feed_dict)
            train_loss = session.run([mod.cross_entropy, mod.l2_loss], feed_dict=feed_dict)
            print('Step: %d, Learning rate: %f, Train loss:' % (i, lr), train_loss)
            loss_curve.append(train_loss[0])
            # test~~
            if i % 100 == 0:
                batch_images_test, batch_labels_test = read_data_test(path=config.test_dir, num_id=train_num_id_test, tongji=tongji_test,
                                                                        image_width=mod.image_width, image_height=mod.image_height,
                                                                        batch_size=mod.batch_size)
                batch_data_test = np.zeros([batch_images_test.shape[1] * 2, batch_images_test.shape[2],
                                            batch_images_test.shape[3], batch_images_test.shape[4]])
                batch_data_test[::2, :, :, :] = batch_images_test[0]
                batch_data_test[1::2, :, :, :] = batch_images_test[1]
                feed_dict_test = {mod.images: batch_data_test,
                                  mod.labels: batch_labels_test, mod.training: False}
                test_loss = session.run([mod.cross_entropy, mod.l2_loss, mod.accuracy, mod.recall, mod.precision], feed_dict=feed_dict_test)
                print("Testing Epoch: %d ..." % (i + 1))
                print('Step: %d, Learning rate: %f, Test loss:' % (i, lr), test_loss)
            if (i+1) % 1000 == 0:
                model_saver.save(session, config.output_dir + 'model_visual.ckpt', i)
        print(loss_curve)
