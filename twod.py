from keras.models import load_model
from keras import backend as k
import numpy as np
import pickle
from pyimage import bin_count_Centrist
from time import time
from keras.layers import Dense,Convolution1D,Input,LSTM
from keras.models import Model

""" This is code for calculating centrist using faster version """
def load_ped_centrist(x):
    """ :returns x , y
        where x is an array of centrist vectors
        and y is an array of labels """
    x_centrist = []
    print "Calculating centrist transforms..."
    t = time()
    for image in x:
        ct = bin_count_Centrist(image)
        x_centrist.append(ct)
    print "Centrist took {0}s for {1} files".format(round(time() - t), len(x))
    x_centrist = np.array(x_centrist)
    return x_centrist
"Loading the pretraine model..."

# model = load_model('twod_model.h5')
#
# "This is the code for feature extraction"
# def get_attributes(model,layer_idx,x_batch):
#     get_attributes =k.function([model.layers[0].input,k.learning_phase()],[model.layers[layer_idx].output])
#     activation = get_attributes([x_batch,0])
#     return activation
#
# with open('daria.pickle', 'rb') as data:
#     data_new = pickle.load(data)
#
# def feature_ext(input_data,prop_model,seq_size,layer_number,number_param):
#     l = time()
#     list = []
#     for i in xrange(int(len(input_data)/seq_size)):
#         j = i * seq_size
#         data = input_data[j:j+seq_size]
#
#         t = time()
#         print "\nPerforming extraction task {0}.....".format(i+1)
#
#         act = get_attributes(prop_model,layer_number,data)
#         "1280 is the layer size at the 11 layer in the proposed 2d model"
#
#         act = np.array(act)
#         print act.shape[2]
#         act = np.reshape(act,(seq_size,number_param))
#         print "Extraction 2d for task {0} took...".format(i+1),time()-t,"sec"
#         "Centrist append here"
#         x = load_ped_centrist(input_data[j:j+seq_size])
#         # concat = np.hstack((act,x)).reshape(1, x.shape[0], -1)
#         concat = np.hstack((act, x))
#         list.append(concat)
#     print '\nNummber of Samples generated:',len(list)
#     print 'Total time for 2d Extraction and Centrist append:',time()-l
#     array = np.array(list)
#     return array, seq_size , array.shape[2]
#
# ext_list, seq ,param = feature_ext(data_new,model,10,11,1280)
#
# h = time()
#
#
#
# def lstm_net(seq_size,param):
#     "This is the 2d network...."
#
#     inputs_1d = Input(shape=(seq_size,param))
#
#     _dconv = Convolution1D(nb_filter=100,filter_length=3,activation='sigmoid',border_mode='same')(inputs_1d)
#     lstm = LSTM(10)(_dconv)
#     # flat = Flatten()(_dconv)
#     dense_lstm = Dense(1,activation='sigmoid')(lstm)
#
#     model = Model(input=inputs_1d, output=dense_lstm,name='model_1')
#     model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
#     # print "Summary of the Model:", (model.summary())
#
#     return model
#
# model_lstm = lstm_net(seq,param)
# pred= model_lstm.predict(ext_list,verbose=1)
#
# result=[]
# for _ in pred:
#     if _ > 0.5:
#         result.append(1)
#     else:
#         result.append(0)
# print result
#
# print "time taken by lstm network",time()-h,"sec..."
#
# print "Showing on the video..."
#
# import cv2
#
# cap = cv2.VideoCapture('daria_jump.avi')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow('test',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
