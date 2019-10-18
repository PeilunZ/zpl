import tensorflow as tf
import numpy as np
import os
import re
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

print('load BS-C')

len_margin = 0
#######################################################################
## shakespear
sharpath = '/home/zpl/danmu_big.json'
file = open(sharpath, 'r', encoding='utf-8')
all = file.readlines()
for i in range(len(all)):
    all[i] = json.loads(all[i])
file.close()
all = [x['text'] for x in all if x['text'] != '\n']
all = [x for x in all if len(x) < 21 and len(x) > len_margin]
#for x in range(len(all)):
#    all[x] = all[x][:-1] + ' ' + all[x][-1]
#######################################################################
basepath = '/home/zpl/comic_person/new_project'
zimu_path = basepath + '/' + 'whole_zimu_voc_1000.txt'
danmu_path = basepath + '/' + 'whole_danmu_voc_1000.txt'

def vocab_take(voc_path):
    vocab = {}
    file = open(voc_path, 'r', encoding='utf-8')
    vocab_all = file.readlines()
    for x in vocab_all:
        index = [i for i, j in enumerate(x) if j == ':'][-1]
        vocab[x[0:index]] = int(x[(index + 1):-1])
    file.close()
    return vocab

#vocab_zimu = vocab_take(zimu_path)
vocab = vocab_take(danmu_path)
#print(len(vocab_zimu))
print(len(vocab))
## load done!

index_list = {}
for k in vocab.keys():
    index_list[vocab[k]] = k

class Config():
    vocab_size = len(vocab) + 1
    learning_rate = 0.001
    init_scale = 0.04
    num_layers = 3
    num_steps = 20  # number of steps to unroll the RNN for
    hidden_size = 500  # size of hidden layer of neurons
    iteration = 27
    keep_prob = 0.5
    batch_size = 64
    is_sample = True
    is_beams = True
    beam_size = 2
    model_path = '/home/zpl/comic_person/new_model/lstm_danmu_yuxun/model_longtime'  # the path of model that need to save or load
    run_size = 256
    max_grad_norm = 15
    save_freq = 4
    # parameters for generation
    save_time = 36  # load save_time saved models
    len_of_generation = 500  # The number of characters by generated
    start_sentence = '完'  # the seed sentence to generate text

Config = Config()

class RNN():
    def __init__(self, config):
        self.run_size = config.run_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.hidden_size = config.hidden_size
        # self.lr = config.learning_rate
        self.max_grad_norm = config.max_grad_norm
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.run_size, forget_bias=0.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=Config.keep_prob)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * Config.num_layers, state_is_tuple=True)
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.target_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        embedding = tf.Variable(tf.truncated_normal([self.vocab_size, self.run_size], stddev=0.1), name='embedding')
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        outputs = []
        softmax_w = tf.get_variable('softmax_w', [self.run_size, self.vocab_size])
        softmax_b = tf.get_variable('softmax_b', [self.vocab_size])
        with tf.variable_scope('RNN'):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if time_step == 0:
                    output, state = self.cell(inputs[:, time_step, :], self.initial_state)
                else:
                    output, state = self.cell(inputs[:, time_step, :], state)
                outputs.append(output)
        self.final_state = state
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.run_size])
        self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.prob = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.target_data, [-1])],
            [tf.ones([self.batch_size * self.num_steps])])
        self._cost = tf.reduce_sum(loss) / self.batch_size
        self.lr = tf.placeholder(dtype=tf.float32, shape=None)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr[0])
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

print('transfer to raw data')

raw = []
count = 0
tian = vocab[' ']
left_space = Config.num_steps
for x in all:
    count = count + 1
    x_app = []
    if len(x) > left_space:
        space = [tian] * left_space
        raw.extend(space)
        left_space = Config.num_steps
    for y in list(x):
        if y in vocab.keys():
            x_app.append(vocab[y])
        else:
            x_app.append(Config.vocab_size - 1)
    raw.extend(x_app)
    left_space = left_space - len(x)
    if left_space != 0:
        raw.append(tian)
        left_space = left_space - 1

print(raw[-30:])
print('transfer done')
#####################################################################
## for shakespear
#raw = []
#count = 0
#vocab['1'] = Config.vocab_size - 1
#for x in all:
#    count = count + 1
#    x_app = []
#    for y in x.split(' '):
#        if y in vocab.keys():
#            x_app.append(vocab[y])
#        else:
#            x_app.append(Config.vocab_size)
#    raw.extend(x_app)
#    if count % 100000 == 0:
#        print(count)
#####################################################################
#file = open('F:\\raw_barrage.txt', 'r')
#raw = file.readline()

def data_iterator(raw_data):      # raw_data:raw
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    num_steps_new = Config.num_steps + 1
    batch_len = data_len // Config.batch_size
    epoch_size = (batch_len - 1) // num_steps_new
    #batch_size_new = data_len // (num_steps_new * epoch_size)
    #raw_data = raw_data[:(num_steps_new * epoch_size) * batch_size_new]
    data = np.zeros([Config.batch_size, (num_steps_new * epoch_size)], dtype=np.int32)
    for i in range(Config.batch_size):
        data[i] = raw_data[(num_steps_new * epoch_size) * i:(num_steps_new * epoch_size) * (i + 1)]
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i*Config.num_steps:(i+1)*Config.num_steps]
        y = data[:, i*Config.num_steps+1:(i+1)*Config.num_steps+1]  # y就是x的错一位，即下一个词
        yield (x, y)

def run_epoch(session, m, data, eval_op, iter):
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    m.initial_state = tf.convert_to_tensor(m.initial_state)
    state = m.initial_state.eval()
    #m.initial_state_bw = tf.convert_to_tensor(m.initial_state_bw)
    #state_bw = m.initial_state_bw.eval()

    for step, (x, y) in enumerate(data_iterator(data)):
        #cost, state_fw, state_bw, _, lr = session.run([m._cost, m.final_state_fw, m.final_state_bw, eval_op, m.lr],  # x和y的shape都是(batch_size, num_steps)
        #                             {m.input_data: x,
        #                              m.target_data: y,
        #                              m.initial_state_fw: state_fw,
        #                              m.initial_state_bw: state_bw,
        #                              m.lr: np.array([Config.learning_rate * (0.9 ** iter)])})
        cost, state, _, lr = session.run([m._cost, m.final_state, eval_op, m.lr],# x和y的shape都是(batch_size, num_steps)
                                                      {m.input_data: x,
                                                       m.target_data: y,
                                                       m.initial_state: state,
                                                       m.lr: np.array([Config.learning_rate * (0.99 ** iter)])})
        costs += cost
        iters += m.num_steps
        if step and step % (epoch_size // 10) == 0:
            print(lr)
            print("%.2f perplexity: %.3f cost-time: %.2f s" %
                  (step * 1.0 / epoch_size, (costs / iters),
                   (time.time() - start_time)))
            start_time = time.time()
    return np.exp(costs / iters)

with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-Config.init_scale,
                                                Config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        mod = RNN(config=Config)
    tf.global_variables_initializer().run()
    model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    for i in range(Config.iteration):
        print("Training Epoch: %d ..." % (i + 1))
        train_perplexity = run_epoch(session, mod, raw, mod._train_op, i)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        if (i + 1) % Config.save_freq == 0:
            print('model saving ...')
            model_saver.save(session, Config.model_path + '-%d' % (i + 1))
            print('Done!')



# 

