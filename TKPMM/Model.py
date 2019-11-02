import numpy as np
import json
import os
import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
# import matplotlib.image as mpimg
from tensorflow import pywrap_tensorflow
import re
import random
import tensorflow.contrib.slim as slim
import jieba
import heapq

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
    run_size = 256
    run_size_z = 300
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    tem = 1.0
    batch_size = 50
    lr = 0.003
    image_height = 256
    image_width = 128
    weight_decay = 0.0005
    init_scale = 0.04
    max_grad_norm = 15
    momentum = 0.9  #
    epoches = 5000
    data_dir = '/home/zpl/qikan/train_ban/train_ban'
    mode = 'train'
    load_dir = '/home/zpl/comic_person/output_new_random_shoulian2/'
    test_dir = '/home/zpl/qikan/test_ban/test_ban'
    gallery_dir = '/home/zpl/qikan/test_ban/test_ban'
    output_new = '/home/zpl/comic_person/output_10yuehou_expert_final_2_10_45_mixed/'
    period_before = -10
    period_after = 15
    period_num = 5
    period_dis = (period_after - period_before) / period_num
    period_before_z = -45
    period_after_z = 45
    period_num_z = 10
    period_dis_z = (period_after_z - period_before_z) / period_num_z

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
            # filter_shape = [kernel_size, kernel_size, n_channels, n_channels]  # Q: why "n_channels" filter?
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
        upsample_shape = self.resnetlayer_shape[resnet_layer_index - 1]  # last layer:3
        upscore_name = "upscore2_" + str(resnet_layer_index)
        feature_deconv = self.upsample_layer(bottom_map=input, name=upscore_name, upscale_factor=2,
                                             shape=upsample_shape)
        feature_deconv = self.batch_norm(feature_deconv, self.training, self.data_format)
        feature_deconv = tf.nn.relu(feature_deconv)
        filling_map = tf.layers.conv2d(
            inputs=self.resnet_feature[resnet_layer_index - 1], filters=upsample_shape[-1], kernel_size=1, strides=1,
            padding='SAME', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=self.data_format)
        upsample_map = tf.add(feature_deconv, filling_map)
        return upsample_map

    def semantic(self, img):  # input:[50, 8, 4, 512]
        img_sqr = tf.add(tf.multiply(img, img), img)
        img_global = tf.reduce_mean(input_tensor=img_sqr, axis=[1, 2])  # [50, 512]
        fc_text = tf.expand_dims(input=tf.layers.dense(img_global, 128, tf.nn.relu,
                                                       name='fc_to_text', reuse=tf.AUTO_REUSE), axis=-1)
        return fc_text  # [50, 128, 1]

    def semantic_expert(self, img1, img2):
        img1_sqr = tf.add(tf.multiply(img1, img1), img1)
        img1_global = tf.reduce_mean(input_tensor=img1_sqr, axis=[1, 2])
        img2_sqr = tf.add(tf.multiply(img2, img2), img2)
        img2_global = tf.reduce_mean(input_tensor=img2_sqr, axis=[1, 2])
        img_global = tf.expand_dims(tf.concat([tf.expand_dims(img1_global, 1), tf.expand_dims(img2_global, 1)], 1),
                                    -1)  # [50, 2, 512, 1]
        fc_text_e1 = tf.layers.conv2d(
            inputs=img_global, filters=1, kernel_size=2, strides=2,
            padding='VALID', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format='channels_last', name='fc_to_text_e1')
        fc_text_e2 = tf.expand_dims(input=tf.layers.dense(tf.reshape(fc_text_e1, [self.batch_size, 256]), 2, tf.nn.relu,
                                                          name='fc_to_text_e2', reuse=tf.AUTO_REUSE), axis=-1)
        return fc_text_e2

    def semantic_z(self, img):  # input:[50, 8, 4, 512]
        img_sqr = tf.add(tf.multiply(img, img), img)

        img_global = tf.reduce_mean(input_tensor=img_sqr, axis=[1, 2])  # [50, 512]
        # img_semantic = tf.layers.batch_normalization(
        #    inputs=img_global, axis=1,
        #    momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
        #    scale=True, training=self.training, fused=True)
        fc_text_z = tf.expand_dims(input=tf.layers.dense(img_global, 128, tf.nn.relu,
                                                         name='fc_to_text_z', reuse=tf.AUTO_REUSE), axis=-1)
        return fc_text_z

    def encode(self, text):
        text_shape = text.get_shape().as_list()
        text_pre = tf.reshape(text, [text_shape[0] * text_shape[1], -1])
        fc_embedding = tf.reshape(tf.layers.dense(text_pre, 128, tf.nn.relu, name='fc_embedding', reuse=tf.AUTO_REUSE),
                                  [text_shape[0], text_shape[1], 128])
        return fc_embedding

    def encode_zimu(self, text):
        text_shape = text.get_shape().as_list()
        text_pre = tf.reshape(text, [text_shape[0] * text_shape[1], -1])
        fc_embedding = tf.reshape(
            tf.layers.dense(text_pre, 128, tf.nn.relu, name='fc_embedding_z', reuse=tf.AUTO_REUSE),
            [text_shape[0], text_shape[1], 128])
        return fc_embedding

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.text_num = config.period_num
        self.text_num_z = config.period_num_z
        self.run_size = config.run_size
        self.run_size_z = config.run_size_z
        self.weight_decay = config.weight_decay
        self._BATCH_NORM_DECAY = config._BATCH_NORM_DECAY
        self._BATCH_NORM_EPSILON = config._BATCH_NORM_EPSILON
        self.images = tf.placeholder(tf.float32, [self.batch_size * 2, self.image_height, self.image_width, 3],
                                     name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 2], name='labels')
        self.text_x_z = tf.placeholder(tf.float32, [self.batch_size, self.text_num_z, self.run_size_z])
        self.text_y_z = tf.placeholder(tf.float32, [self.batch_size, self.text_num_z, self.run_size_z])
        self.text_x = tf.placeholder(tf.float32, [self.batch_size, self.text_num, self.run_size])
        self.text_y = tf.placeholder(tf.float32, [self.batch_size, self.text_num, self.run_size])
        self.lr = tf.placeholder(dtype=tf.float32, shape=None)
        self.block_fn = self._building_block_v1
        self.data_format = config.data_format
        self.num_filters = config.num_filters  # The number of filters to use for the first block layer
        self.kernel_size = config.kernel_size  # kernel_size: The kernel size to use for convolution.
        self.conv_stride = config.conv_stride
        self.block_sizes = config.block_sizes
        self.pre_activation = config.pre_activation
        self.block_strides = config.block_strides  # List of integers representing the desired stride size for
        self.training = tf.placeholder(tf.bool)
        self.isrichang = tf.placeholder(tf.bool)
        self.first_pool_size = config.first_pool_size
        self.first_pool_stride = config.first_pool_stride
        self.tem = config.tem
        self.dtype = tf.float32
        self.momentum = config.momentum
        self.resnetlayer_shape = {}  # image shape after each block
        self.resnet_feature = {}  # the output of the 0,1,2,3,4-th layer
        self.upsample_feature = {}
        self.before_global = {}
        self.dtype = tf.float32
        images = tf.identity(self.images, 'origin_inputs')
        inputs = tf.cast(images, dtype=self.dtype)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format)
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

        x_to_semantic = self.semantic(inputs[::2, :, :, :])
        x_to_semantic_z = self.semantic_z(inputs[::2, :, :, :])
        y_to_semantic = self.semantic(inputs[1::2, :, :, :])
        y_to_semantic_z = self.semantic_z(inputs[1::2, :, :, :])
        expert = self.semantic_expert(inputs[::2, :, :, :], inputs[1::2, :, :, :])  # [50,2]
        expert_d = tf.tile(tf.expand_dims(expert[:, 0], -1), [1, 5, 128])
        expert_z = tf.tile(tf.expand_dims(expert[:, 1], -1), [1, 10, 128])

        embedding_x_d = self.encode(self.text_x)
        embedding_y_d = self.encode(self.text_y)
        embedding_x_z = self.encode_zimu(self.text_x_z)
        embedding_y_z = self.encode_zimu(self.text_y_z)

        x_semantic_attention_d = tf.nn.softmax(
            tf.reshape(tf.matmul(embedding_x_d, x_to_semantic), [self.batch_size, 1, -1]))  # [50, 1, 5]
        x_semantic_attention_d_tile = tf.transpose(tf.tile(x_semantic_attention_d, [1, 128, 1]),
                                                   [0, 2, 1])  # [50, 5, 128]
        x_attention_text_d = tf.multiply(x_semantic_attention_d_tile, embedding_x_d)  # [50, 5, 128]

        y_semantic_attention_d = tf.nn.softmax(tf.reshape(tf.matmul(embedding_y_d, y_to_semantic),
                                                          [self.batch_size, 1, -1]))
        y_semantic_attention_d_tile = tf.transpose(tf.tile(y_semantic_attention_d, [1, 128, 1]),
                                                   [0, 2, 1])  # [50, 5, 128]
        y_attention_text_d = tf.multiply(y_semantic_attention_d_tile, embedding_y_d)  # [50, 5, 128]
        # warping substracting

        matching_temp_d = tf.matmul(x_attention_text_d, tf.transpose(y_attention_text_d, [0, 2, 1]))
        diff_text_wrap_d = tf.multiply(x_attention_text_d - tf.matmul(matching_temp_d, y_attention_text_d),
                                       expert_d)  # [50, 5, 128]

        # done!
        x_semantic_attention_z = tf.nn.softmax(
            tf.reshape(tf.matmul(embedding_x_z, x_to_semantic_z), [self.batch_size, 1, -1]))
        x_semantic_attention_z_tile = tf.transpose(tf.tile(x_semantic_attention_z, [1, 128, 1]), [0, 2, 1])
        x_attention_text_z = tf.multiply(x_semantic_attention_z_tile, embedding_x_z)  #

        y_semantic_attention_z = tf.nn.softmax(tf.reshape(tf.matmul(embedding_y_z, y_to_semantic_z),
                                                          [self.batch_size, 1, -1]))
        y_semantic_attention_z_tile = tf.transpose(tf.tile(y_semantic_attention_z, [1, 128, 1]), [0, 2, 1])
        y_attention_text_z = tf.multiply(y_semantic_attention_z_tile, embedding_y_z)  # [50, 10, 128]

        matching_temp_z = tf.matmul(x_attention_text_z, tf.transpose(y_attention_text_z, [0, 2, 1]))
        diff_text_wrap_z = tf.multiply(x_attention_text_z - tf.matmul(matching_temp_z, y_attention_text_z),
                                       expert_z)  # [50, 10, 128]
        ###  union
        semantic_dis_d = tf.reshape(tf.reduce_mean(tf.reshape(diff_text_wrap_d, [self.batch_size, 4, 32, -1]), 1),
                                    [self.batch_size, -1])  # [50, 32*5]

        semantic_dis_z = tf.reshape(tf.reduce_mean(tf.reshape(diff_text_wrap_z, [self.batch_size, 4, 32, -1]), 1),
                                    [self.batch_size, -1])  # [50, 32*10]
        # print(semantic_dis.get_shape())
        self.semantic_dis_d = tf.layers.batch_normalization(
            inputs=semantic_dis_d, axis=1,
            momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.training, name='semantic_bn_d', fused=True)

        self.semantic_dis_z = tf.layers.batch_normalization(
            inputs=semantic_dis_z, axis=1,
            momentum=self._BATCH_NORM_DECAY, epsilon=self._BATCH_NORM_EPSILON, center=True,
            scale=True, training=self.training, name='semantic_bn_z', fused=True)

        self.semantic_dis = tf.concat([self.semantic_dis_d, self.semantic_dis_z], 1)

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
            # diff_attention_layer_map = tf.add(tf.multiply(diff_map, attention_layer_map), diff_map)
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

        self.last_map = tf.concat([self.temp[-1], self.semantic_dis], 1)
        for i in range(0, len(self.block_sizes) - 1):
            self.last_map = tf.concat([self.last_map, self.temp[i]], 1)  # [batch_size, 512 + 256 + 64 + 32 + 16 + 8]
        fc1 = tf.layers.dense(self.last_map, 100, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')
        self.logits = fc2
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        tf.identity(self.cross_entropy, name='cross_entropy')
        # precision&recall&accuracy
        precision_pre = tf.nn.softmax(self.logits)

        self.soft_logit = tf.nn.softmax(self.logits)
        pre_thres = precision_pre[:, 0]
        pre_thres_2 = precision_pre[:, 1]
        thres = 0.30
        self.precision_thres = {}
        self.recall_thres = {}
        self.accuracy_thres = {}
        for i in range(5):
            comp = [thres] * list(pre_thres.get_shape())[0]
            compp = tf.cast(pre_thres > comp, tf.int64)
            compp_2 = tf.cast(pre_thres_2 > comp, tf.int64)
            self.accuracy_thres[thres] = (tf.reduce_sum(
                tf.cast(tf.logical_and(tf.cast(compp, tf.bool), tf.cast(tf.argmin(self.labels, 1), tf.bool)),
                        tf.float32)) + tf.reduce_sum(
                tf.cast(tf.logical_and(tf.cast(compp_2, tf.bool), tf.cast(tf.argmax(self.labels, 1), tf.bool)),
                        tf.float32))) / self.batch_size
            if tf.reduce_sum(compp) == 0.0:
                self.precision_thres[thres] = 0.0
            else:
                self.precision_thres[thres] = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(compp, tf.bool),
                                                                                   tf.equal(tf.argmax(self.labels, 1),
                                                                                            tf.zeros([self.batch_size],
                                                                                                     tf.int64))),
                                                                    tf.int64)) / tf.reduce_sum(compp)
            if tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.zeros([self.batch_size], tf.int64)),
                                     tf.int64)) == 0.0:
                self.recall_thres[thres] = 0.0
            else:
                self.recall_thres[thres] = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(compp, tf.bool),
                                                                                tf.equal(tf.argmax(self.labels, 1),
                                                                                         tf.zeros([self.batch_size],
                                                                                                  tf.int64))),
                                                                 tf.int64)) / tf.reduce_sum(
                    tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.zeros([self.batch_size], tf.int64)), tf.int64))
            thres = round(thres + 0.1, 1)

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
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.lr,
            momentum=self.momentum)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train = optimizer.apply_gradients(zip(grads, tvars))
