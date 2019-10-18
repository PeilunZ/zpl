#pylint: skip-file
import os
cudaid = 2
os.environ["THEANO_FLAGS"] = "device=cuda" + str(cudaid)

import time
import sys
import numpy as np
import theano
import theano.tensor as T
from NTM import *
import data
import json
#import matplotlib.pyplot as plt

#use_gpu(2)

lr = 0.005
drop_rate = 0.2
batch_size = 20
hidden_size = 500
latent_size = 100
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "momentum"
continuous = False

train_idx, valid_idx, test_idx, other_data = data.load("/home/zpl/neural-topic-model-master/barrage_one.txt") #movie.txt, yelp.txt
[docs, dic, w2i, i2w, bg] = other_data
#file = open('/home/zpl/neural-topic-model-master/barrage_vocab.json', 'w', encoding='utf-8')
#file.write(json.dumps(w2i))
#file.close()

print(w2i)
#for key, value in sorted(dic.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)


dim_x = len(dic)
dim_y = dim_x
print("#features = ", dim_x, "#labels = ", dim_y)

print("compiling...")
model = NTM(dim_x, dim_x, hidden_size, latent_size, bg, continuous, optimizer)

print("training...")
start = time.time()
for i in range(50):
    train_xy = data.batched_idx(train_idx, batch_size)
    error = 0.0
    e_l1 = 0.0
    e_nll = 0.0
    e_kld = 0.0
    e_cre = 0.0
    in_start = time.time()
    for batch_id, x_idx in train_xy.items():
        X = data.batched_news(x_idx, other_data)
        cost, cre, kld, nll, l1, z = model.train(X, lr)
        error += cost
        e_l1 += l1
        e_nll += nll
        e_kld += kld
        e_cre += cre
        #print i, batch_id, "/", len(train_xy), cost
    in_time = time.time() - in_start

    error /= len(train_xy)
    e_l1 /= len(train_xy)
    e_nll /= len(train_xy)
    e_kld /= len(train_xy)
    e_cre /= len(train_xy)
    print("Iter = " + str(i) + ", Loss = " + str(error) + ", cre = " + str(e_cre) + ", kld = " + str(e_kld) + ", nll = " + str(e_nll) + ", L1 = " + str(e_l1) + ", Time = " + str(in_time))

print("training finished. Time = " + str(time.time() - start))

print("save model...")
if not os.path.exists("/home/zpl/neural-topic-model-master/output_model_100/"):
    os.makedirs("/home/zpl/neural-topic-model-master/output_model_100/")
save_model("/home/zpl/neural-topic-model-master/output_model_100/vae_text.model", model)

print("lode model...")
load_model("/home/zpl/neural-topic-model-master/output_model_100/vae_text.model", model)

print("validation..")
valid_xy = data.batched_idx(valid_idx, batch_size)
error = 0.0
e_nll = 0.0
for batch_id, x_idx in valid_xy.items():
    X = data.batched_news(x_idx, other_data)
    cost, cre, kld, nll, y = model.validate(X)
    error += cost
    e_nll += nll
print("Loss = " + str(error / len(valid_xy)), ", NLL = ", e_nll / len(valid_xy))

top_w = 100
## manifold 
if latent_size == 2:
    test_xy = data.batched_idx(test_idx, 1000)
    x_idx = test_xy[0]
    X = data.batched_news(x_idx, other_data)

    mu = np.array(model.project(X))
    
    #plt.figure(figsize=(8, 6))
    #plt.scatter(mu[:, 0], mu[:, 1], c="r")
    #plt.savefig("2dstructure.png", bbox_inches="tight")
    #plt.show()

    nx = ny = 20
    v = 100
    x_values = np.linspace(-v, v, nx)
    y_values = np.linspace(-v, v, ny) 
    canvas = np.empty((28*ny, 20*nx))
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)[0,:]
            ind = np.argsort(-y)
            print(xi, yi)
            for k in range(top_w):
                print(i2w[ind[k]])
            print("\n")
else:
    sampels = 10
    #for i in range(sampels):
    #    z = model.noiser(latent_size)
    #    y = model.generate(z)[0,:]
    #    ind = np.argsort(-y)
    #    for k in range(top_w):
    #        print(i2w[ind[k]])
    #    print("\n")
#
    #print("\n\n")
    weights = np.array(model.W_hy.get_value())
    for i in range(latent_size):
        ind = list(np.argsort(weights[i, :]).tolist())
        ind.reverse()
        print("Topic #" + str(i+1) + ": ")
        doc2word = ''
        for k in range(top_w):
            doc2word = doc2word + i2w[ind[k]]
        print(doc2word)
    
    #print("\n\n")
    #b = np.array(model.b_hy.get_value())
    #ind = list(np.argsort(b).tolist())
    #ind.reverse()
    #for k in range(top_w):
    #    print(i2w[ind[k]])
    #print("\n")
    file = open('/home/zpl/vae_topic_model_100/embedding_vae.json', 'w')
    weights_dic = {}
    for i in range(weights.shape[1]):
        weights_dic[i] = list(weights[:,i])

    file.write(json.dumps(weights_dic))
    file.close()
