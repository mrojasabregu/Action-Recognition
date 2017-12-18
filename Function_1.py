from keras.models import load_model
import cv2
import glob
import numpy as np
import os
from os.path import join,isfile
import scipy.misc
import pickle
from keras import backend as k
from keras.models import Sequential
from keras.layers import Convolution2D,Activation
import theano
from keras.utils import np_utils

path = 'C:/Users/USER/Desktop/Downloaded DataSets/Weizmann Action DataSet'
# model = load_model('Wiseman_model.h5')
# model1 = load_model('Wiseman_one_d1.h5')
# from time import time
# with open('train.pickle', 'rb') as data:
#     data_new = pickle.load(data)
# data_new = data_new[130:140]
# # # print data_new
# get_activations = k.function([model.layers[0].input], model.layers[7].output)
# t = time()
# d =get_activations([data_new]).reshape(1,10,192)
#
# # # print d
# p = model1.predict(d,verbose=1)
# print time()-t
# print np.argmax(p)

# def test(path):
#
#         first_file = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]



# path = 'C:\Users\USER\Desktop\Downloaded DataSets\Weizmann Action DataSet'

def myGenerator(path):
    Y_train=[]
    X_train=[]
    list = os.listdir(path)

    for i in xrange(len(list)):
         list_dir = glob.glob(path + str('/') + str(list[i]) + str('/*'))
         total_count =0
         for _ in list_dir:
            cap = cv2.VideoCapture(_)
            # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            back_sep = cv2.createBackgroundSubtractorKNN()
            count = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if(ret == True):
                 frame =np.array(back_sep.apply(frame))

                 retu , thr = cv2.threshold(frame, 0, 136, cv2.THRESH_BINARY )
                 # print thr.shape
                 frame = scipy.misc.imresize(thr,(64,64,1))

                 # print np.array(frame).shape
                 # frame = plt.imread(frame)
                 # plt.imshow(frame)
                 # plt.show()
                 # cv2.imshow('frame',frame)
                 # k = cv2.waitKey(30) & 0xff
                 # if k ==27:
                 #  break
                if ret == False:
                    print "Number of Frames Extracted :", count
                    break
                count = count + 1
                X_train.append(frame)


            total_count = total_count + count

         for j in xrange(total_count):
             Y_train.append(i)


         print 'Directory ${0}$ Extracted with total frames : {1}\n'.format(list[i],total_count)

    print ('Sucessfully Extracted {0} frames with {1} labels'.format(len(X_train),len(Y_train)))

    Y_train = np_utils.to_categorical(Y_train,len(list))
    X_train = np.array(X_train).reshape(4676,64,64,1)
    # X_train = np.array(X_train)
    # if yield_bool == False:
    return X_train, Y_train


# c,t= myGenerator(path,10)
# print c.shape[0]
# while 1:
#     print c.next()
# t = np_utils.categorical_probas_to_classes(t)
# count , unique = np.unique(t,return_counts=True)
# print count , sum(unique)
#

# d = c[0:10]
# x= c[10:20]
# from time import time
# # tran = np.reshape(tran,())
# print tran.shape

# l = time()
# get_activations([d])
# get_activations([x])[0]
# print time()-l
# def get_attributes(model,layer_idx,x_batch):
#     get_attributes =k.function([model.layers[10].input,k.learning_phase()],[model.layers[layer_idx].output])
#     activation = get_attributes([x_batch,0])
#     return activation
# from time import time
# t = time()
# y = get_attributes(model,11,tran)
# print time()-t
# y= np.array(y)
# print y.shape
# y = np.reshape(y,(20,8,8,16))
# y = y[0,:,:,1]
# import matplotlib.pyplot as plt
# plt.imshow(y)
# plt.show()
