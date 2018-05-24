import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from numpy.random import *
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import StratifiedKFold
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, report, training, Chain, datasets, iterators, cuda, Reporter, report_scope
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions  as F
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import sys
import sqlite3
import re
from gensim import corpora, matutils
from gensim.models import word2vec
from collections import defaultdict
import six
import argparse
from PIL import Image

plt.switch_backend('agg')

#テキストデータの読み込み，分割
print('load text ...')
f = open('./Flickr8k_Dataset/Flickr8k_text/Flickr8k.token.txt')
tmp = f.read()
f.close()

tmp2 = [] 
tmp2 = tmp.split('\n')

tmp3 = []
for t in tmp2:
    tmp3.append(t.split('	'))

tmp4 = []
for i in range(len(tmp3)):
    tmp4.append(tmp3[i][0].split('#'))

tmp5 = []
num = -1

for i in range(len(tmp2)):
    if tmp4[i][1] == '0':
        tmp5.append(tmp4[i][0] + '# ' + tmp3[i][1])
        num  += 1
    else:
        tmp5[num] = tmp5[num] + ' ' + tmp3[i][1]

tmp6 = []
for t in tmp5:
    tmp6.append(t.split('#'))

text = []
for i in range(len(tmp6)):
    l = tmp6[i][0]
    if l.endswith('.jpg'):
        text.append(tmp6[i])

max_text = 0
for i in range(len(text)):
    if max_text<len(text[i][1]):
        max_text = len(text[i][1])

print('max length :', max_text, 'words')
print('data :', len(text), ' pairs.')

# Word2Vec
print('start Word2Vec ...')
all_text = ''
for i in range(len(text)):
    all_text = all_text + ' ' + text[i][1]
f = open("./Flickr8k_Dataset/result/all_text.txt", 'w')
f.write(all_text)
f.close()

vector_size = 100 # word2vecの次元数

data = word2vec.Text8Corpus("./Flickr8k_Dataset/result/all_text.txt")
w2v_model = word2vec.Word2Vec(data, size=vector_size)
w2v_model.save("./Flickr8k_Dataset/result/all_text_" +str(vector_size)+".model")
voc = w2v_model.wv.vocab.keys()
voc_num = len(w2v_model.wv.vocab)
wv = []
vocnew = []
for x in voc:
  try:
    wv.append(w2v_model[x])
  except KeyError:
    print (x, 'を無視します')
  vocnew.append(x)

text_x = []
words = []
ari = 0
nashi = 0
t_height = 500 #1つのtextデータで用いる単語数

for i in range(len(text)):
    t_tmp = np.zeros((t_height, vector_size), dtype=float)
    words = text[i][1].split(" ")
    if len(words)>t_height:
        size = t_height
    else :
        size = len(words)
    for j in range(size):
        word = words[j]
        if word in vocnew:
            ari = ari + 1
            t_tmp[j] = w2v_model[word]
        else:
            nashi = nashi + 1
    text_x.append(t_tmp)
print ('ベクトル有り：', ari, '単語')
print('ベクトルなし：', nashi, '単語')

#画像の読み込み
print('load images ...')
max_w = 0
max_h = 0
for i in range(len(text)):
    im = Image.open('./Flickr8k_Dataset/Flicker8k_Dataset/' +str(text[i][0]))
    width, height = im.size
    if width>max_w:
        max_w = width
    if height>max_h:
        max_h = height
print('max width :', max_w)
print('max height :', max_h)

image = []
for i in range(len(text)):
    i_tmp = np.zeros((max_w, max_h), dtype=float)
    im = Image.open('./Flickr8k_Dataset/Flicker8k_Dataset/' +str(text[i][0]))
    im = im.convert('L')
    width, height = im.size
    for y in range(height):
        for x in range(width):
            i_tmp[x][y] = im.getpixel((x,y))
    i_tmp = np.array(i_tmp)
    image.append(i_tmp)

#偶数番目は正ペア，奇数番目は誤ペア
print('data shuffle ...')
image_x = []
t = []
for i in range(len(text)):
    if i%2==0:
        image_x.append(image[i])
        t.append('1')
    if i%2==1:
        if i+2<len(text):
            image_x.append(image[i+2])
        else:
            image_x.append(image[1])
        t.append('0')

#CNNの設定
print('set CNN ...')
class SimpleCNN(Chain):
    
    def __init__(self, input_channel, output_channel, t_filter_height, t_filter_width, i_filter_height):
        super(SimpleCNN, self).__init__(
            conv_t = L.Convolution2D(input_channel, output_channel, (t_filter_height, t_filter_width)),
            conv_i = L.Convolution2D(input_channel, output_channel, i_filter_height),
        )
    
    def __call__(self, xt, xi):
        ht = F.max(F.max(F.relu(self.conv_t(xt)), axis=2), axis=2)
        hi = F.max(F.max(F.relu(self.conv_i(xi)), axis=2), axis=2)
        product = ht*hi
        max_product = F.max(product, axis=1)
        max_ = max_product.reshape(max_product.shape[0], 1)
        y = F.sigmoid(max_)
        y_ = 1 - F.sigmoid(max_)
        Y = F.concat([y, y_], axis = 1)
        return Y

gpu = -1
batchsize = 10   #minibatch size
n_epoch = 50   #エポック数
k_hold = 5

input_channel = 1

#textのパラメータ
t_height = t_height
t_width = vector_size
t_filter_height = 5
#t_pooling_window = t_height+1-t_filter_height

#imageのパラメータ
i_height = 500
i_width = 500
i_filter_height = 10
#i_pooling_window = i_height+1-i_filter_height

#隠れユニットの数
output_channel = 10

#学習とテスト
dataset = {}
dataset['t_source'] = np.array(text_x)
dataset['i_source'] = np.array(image_x)
dataset['target'] = np.array(t)
dataset['t_source'] = dataset['t_source'].astype(np.float32)
dataset['i_source'] = dataset['i_source'].astype(np.float32)
dataset['target'] = dataset['target'].astype(np.int32)

xt_train, xt_test, xi_train, xi_test, t_train, t_test = train_test_split(dataset['t_source'], dataset['i_source'], dataset['target'], test_size=0.2)
N_test = t_test.size  #test data size
N = len(xt_train)        #train data size

#四次元テンソルに変換
imput_channel = 1
xt_train = xt_train.reshape(len(xt_train), input_channel, t_height, t_width) 
xt_test = xt_test.reshape(len(xt_test), input_channel, t_height, t_width)
xi_train = xi_train.reshape(len(xi_train), input_channel, i_height, i_width) 
xi_test = xi_test.reshape(len(xi_test), input_channel, i_height, i_width)

#t_model = SimpleCNN_for_text(input_channel, output_channel, t_filter_height, t_width)
#i_model = SimpleCNN_for_image(input_channel, output_channel, t_filter_height)
model = SimpleCNN(input_channel, output_channel, t_filter_height, t_width, i_filter_height)

if gpu >= 0:
#print('Use GPU mode')
#xp = cuda.cupy         # cupy遅いからとりあえず使わない
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()

#Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(1, n_epoch + 1):
    print('epoch:', epoch)
    perm = np.random.permutation(N)
    sum_loss = 0
    sum_acc = 0
    for i in range(0, N, batchsize):
        Xt = Variable(np.asarray(xt_train[perm[i:i + batchsize]]))
        Xi = Variable(np.asarray(xi_train[perm[i:i + batchsize]]))
        T = Variable(np.asarray(t_train[perm[i:i + batchsize]]))
            
        model.zerograds()
        Y = model(Xt, Xi)
            
        loss = F.softmax_cross_entropy(Y, T)
        acc = F.accuracy(Y, T)   
            
        sum_loss += loss.data * len(Xt)
        sum_acc += acc.data * len(Xt)
            
        loss.backward()
            
        if epoch != 1:
            optimizer.update()
                
    train_loss.append(sum_loss / N)
    train_acc.append(sum_acc / N)
    print(' train loss:', sum_loss / N, ' train acc:', sum_acc / N)
    
    Xt_test = Variable(np.asarray(xt_test))
    Xi_test = Variable(np.asarray(xi_test))
    T_test = Variable(np.asarray(t_test))
        
    y_test = model(Xt_test, Xi_test)
    sum_test_loss = F.softmax_cross_entropy(y_test, T_test) 
    sum_test_acc = F.accuracy(y_test, T_test) 
    test_loss.append(sum_test_loss.data)
    test_acc.append(sum_test_acc.data)
    print(' test loss:', F.softmax_cross_entropy(y_test, T_test), ' test acc:', F.accuracy(y_test, T_test))
print(test_acc[n_epoch-1])

'''
cross_acc_sum += test_acc[n_epoch-1]

        plt.plot(np.arange(len(train_acc)), np.array(train_acc))
        plt.plot(np.arange(len(test_acc)), np.array(test_acc))
        plt.ylim((0,1))
        plt.xlim((0,len(train_acc)))

print('Accuracy(mean)', cross_acc_sum/k_hold)
'''

#結果（accとloss）の出力
plt.plot(np.arange(len(train_acc)), np.array(train_acc))
plt.plot(np.arange(len(test_acc)), np.array(test_acc))
plt.ylim((0,1))
plt.xlim((0,len(train_acc)))
plt.savefig('./Flickr8k_Dataset/result/acc.png')

plt.plot(np.arange(len(train_loss)), np.array(train_loss))
plt.plot(np.arange(len(test_loss)), np.array(test_loss))
plt.ylim((0,1))
plt.xlim((0,len(train_loss)))
plt.savefig('./Flickr8k_Dataset/result/loss.png')
