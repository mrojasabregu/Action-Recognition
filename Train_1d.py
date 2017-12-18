from keras.layers import LSTM,Input,Flatten,Dropout,Dense,MaxPooling1D,Convolution1D
from keras.models import Model
from Functions_2 import piece_wise
from keras import backend as k
from keras.models import load_model
from Functions_2 import myGenerator
seq_size = 10
param = 16
classes =10
index = 11
epochs = 1
nb_example_to_train = 4853
# nb_example_to_test = 1025
reduced_size = 32

model_2d = load_model('weizmann_train_64.h5')
path = 'C:\Users\USER\Desktop\Downloaded DataSets\Weizmann_train'
# path2 = 'C:\Users\USER\Desktop\Downloaded DataSets\Weizmann_test'
# val_data, val_label = myGenerator(path2,reduced_size,nb_example_to_test)
def lstm_net():

    inputs_1d = Input(shape=(seq_size,param))
    _lstm = LSTM(50)(inputs_1d)
    # _lstm2 = LSTM (25)(_lstm)

    # _dconv = Convolution1D(nb_filter=20,filter_length=4,activation='sigmoid',border_mode='same')(inputs_1d)
    # max = MaxPooling1D()(_dconv)
    # _dconv2 = Convolution1D(20,3,activation='relu',border_mode='same')(max)
    # max2 = MaxPooling1D()(_dconv2)
    dp = Dropout(0.3)(_lstm)
    # _deconv3 = Convolution1D(10,3,activation='relu',border_mode='same')(max2)
    # max3= MaxPooling1D()(_deconv3)
    # flat = Flatten()(dp)
    dense = Dense(classes,activation='softmax')(dp)

    model = Model(input=inputs_1d, output=dense,name='model_1')
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    print model.summary()
    return model

def one_net(seq_size,param,classes):
    "This is the 1d network...."

    inputs_1d = Input(shape=(seq_size,param))

    _dconv = Convolution1D(nb_filter=64,filter_length=3,activation='relu',border_mode='same')(inputs_1d)
    max = MaxPooling1D()(_dconv)
    _dconv2 = Convolution1D(64,3,activation='relu',border_mode='same')(max)
    max2 = MaxPooling1D()(_dconv2)
    dp = Dropout(0.2)(max2)
    # _deconv3 = Convolution1D(10,3,activation='relu',border_mode='same')(max2)
    # max3= MaxPooling1D()(_deconv3)
    flat = Flatten()(dp)
    dense = Dense(classes,activation='softmax')(flat)

    model = Model(input=inputs_1d, output=dense,name='model_1')
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


    return model

# model_2 = lstm_net()
# model_2 = one_net(seq_size,param,classes)
# model_2.save('random.h5')
model_2 = load_model('random.h5')
# model_2 = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Weizmann_train_64_9812_epoch4_seq30.h5')
# model_lstm = load_model('WIEZMAN_lstm.h5')
get_attributes = k.function([model_2d.layers[0].input], [model_2d.layers[index].output])
data,target = myGenerator(path,reduced_size)
# class_val = piece_wise(model=model_2d,data= val_data,label=val_label,seq_size=seq_size,layer_idx=index)
class_1d_input = piece_wise(model=model_2d, data=data, label=target,
                               seq_size=seq_size, layer_idx=index)
# for __ in xrange (200):
#     print next(class_1d_input.produce_seq_from_2d(get_attributes))
pred1 = model_2.predict_generator(class_1d_input.produce_seq_from_2d(get_attributes),val_samples=200)
import pickle
# with open('scatter5.pickle', 'wb') as output:
#    pickle.dump(pred1, output)
with open('scatter1.pickle', 'rb') as data:
    preda = pickle.load(data)
with open('scatter2.pickle', 'rb') as data:
    predb = pickle.load(data)
with open('scatter3.pickle', 'rb') as data:
    predc = pickle.load(data)
with open('scatter4.pickle', 'rb') as data:
    predd = pickle.load(data)
with open('scatter5.pickle', 'rb') as data:
    prede = pickle.load(data)
# # print "done"
# pred2 = model_2.predict_generator(class_1d_input.produce_seq_from_2d(get_attributes),val_samples=600)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(preda[:,0],preda[:,1],c='b',marker='s')
ax.scatter(predb[:,0],predb[:,1],c='r',marker='o')
ax.scatter(predc[:,0],predc[:,1],c='g',marker='s')
ax.scatter(predd[:,0],predd[:,1],c='y',marker='o')
ax.scatter(prede[:,0],prede[:,1],c='w',marker='s')

plt.show()

# for _ in xrange (1):
#
#     history = model_2.fit_generator(class_1d_input.produce_seq_from_2d(get_attributes),nb_example_to_train,epochs,verbose=1)
    # print model_2.metrics_names
    # scores = model_2.evaluate_generator(class_val.produce_seq_from_2d(get_attributes),1000)
    # print scores
    # model_2.save('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Weizmann_train_64_9812_epoch5_seq30.h5')
