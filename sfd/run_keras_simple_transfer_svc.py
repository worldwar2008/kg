# -*- coding:utf-8 -*-
import numpy as np

np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
import h5py
import theano
from sklearn.svm import SVC




pp = '../'
color_type = 3

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (224, 224))
    return resized

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # 注释掉全连接层,因为我们不需要这一层,后面的权重系数导入的时候,也是避开了这一层
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # if weights_path:
    #     model.load_weights(weights_path)
    assert os.path.exists(weights_path), "Model weights file not found (see 'weights_path' variable in script)"
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k>=len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False

    f.close()
    print 'model loaded.'
    for l in model.layers:
        l.trainable = False


    return model

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        # 路径名
        path = os.path.join(pp+'rawdata/train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)

    return X_train, y_train


def load_test():
    print('Read test images')
    path = os.path.join(pp+'rawdata/test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    # 小于目标值的最大整数的浮点数
    # 分批读取测试文件,因为test数据量特别大
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir(pp+'cache'):
        os.mkdir(pp+'cache')
    open(os.path.join(pp+'cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join(pp+'cache', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join(pp+'cache', 'architecture.json')).read())
    model.load_weights(os.path.join(pp+'cache', 'model_weights.h5'))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size,
                                                              random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def create_submission(predictions, test_id, loss):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


# The same as log_loss
def mlogloss(target, pred):
    score = 0.0
    for i in range(len(pred)):
        pp = pred[i]
        for j in range(len(pp)):
            prob = pp[j]
            if prob < 1e-15:
                prob = 1e-15
            score += target[i][j] * math.log(prob)
    return -score / len(pred)


def validate_holdout(model, holdout, target):
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)
    # score = model.evaluate(holdout, target, show_accuracy=True, verbose=0)
    # print('Score holdout: ', score)
    # score = mlogloss(target, predictions)
    # print('Score : mlogloss', score)
    return score


cache_path = os.path.join(pp+'cache', 'train-3.dat')

if not os.path.isfile(cache_path):
    train_data, train_target = load_train()
    cache_data((train_data, train_target), cache_path)
else:
    print('Restore train from cache!')
    (train_data, train_target) = restore_data(cache_path)

batch_size = 64
nb_classes = 10
nb_epoch = 2

# input image dimensions
img_rows, img_cols = 224, 224
# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)
train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
train_target = np_utils.to_categorical(train_target, nb_classes)
train_data = train_data.astype('float32')
train_data /= 255
print('Train shape:', train_data.shape)
print(train_data.shape[0], 'train samples')

X_train, X_test, X_holdout, Y_train, Y_test, Y_holdout = split_validation_set_with_hold_out(train_data, train_target,
                                                                                            0.2)
print('Split train: ', len(X_train))
print('Split valid: ', len(X_test))
print('Split holdout: ', len(X_holdout))

origin_model = read_model()
outputs_dict = dict([(layer.name, layer.output) for layer in origin_model.layers])
inputs_dict = dict([(layer.name, layer.input) for layer in origin_model.layers])
print origin_model.layers[0].input
print "weights"
print origin_model.layers[0].get_weights()
print outputs_dict["convolution2d_11"]
#print origin_model.layers[11].output(train=False)
get_feathure = theano.function([origin_model.layers[0].input],origin_model.layers[11].output,
                               allow_input_downcast=True)
print "feature train"
train_feathure = get_feathure(X_train)
test_feathure = get_feathure(X_test)

svc(train_feathure,Y_train,test_feathure,Y_test)
