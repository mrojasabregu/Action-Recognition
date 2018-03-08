from keras.layers import LSTM,Input,Flatten,Dropout,Dense,MaxPooling1D,Convolution1D
from keras.models import Model
from Functions_2 import piece_wise
from keras import backend as k
from keras.models import load_model
import pickle
seq_size = 20;" sequence size of the the videos to be taken into convolution operation for extracting the temporal information"
param = 16
classes =2
index = 11;"the layer number from which vectors representing frames are extracted from the saved 2D-CNN model "
epochs = 1
nb_example_to_train = 2400
reduced_size = 32

model_2d = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/2D-Cnn\opticalfight32.h5')
path = 'C:\Users\USER\Desktop\Fight data set'
"Lstm implementaion for the classification of sequence"
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
"One dimensional CNN implementation for classification of sequences "
def one_net(seq_size,param,classes):
    inputs_1d = Input(shape=(seq_size,param))

    _dconv = Convolution1D(nb_filter=10,filter_length=3,activation='relu',border_mode='same')(inputs_1d)
    max = MaxPooling1D()(_dconv)
    _dconv2 = Convolution1D(16,3,activation='relu',border_mode='same')(max)
    max2 = MaxPooling1D()(_dconv2)
    dp = Dropout(0.2)(max2)
    # _deconv3 = Convolution1D(10,3,activation='relu',border_mode='same')(max2)
    # max3= MaxPooling1D()(_deconv3)
    flat = Flatten()(dp)
    dense = Dense(classes,activation='softmax')(flat)

    model = Model(input=inputs_1d, output=dense,name='model_1')
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


    return model


model_2 = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/1D-CNN/1dfight32.h5')

get_attributes = k.function([model_2d.layers[0].input], [model_2d.layers[index].output]);"Extract vectors from the layers of" \
                                                                                         "2D-CNN model"

with open('fightdata.pickle', 'rb') as data:
    data = pickle.load(data)
with open('fightlabel.pickle', 'rb') as target:
    target = pickle.load(target)

class_1d_input = piece_wise(model=model_2d, data=data, label=target,
                               seq_size=seq_size, layer_idx=index);"generate the sequences one-by-one along with labels to be fed " \
                                                                   "to a 1D-CNN model"
"Train the 1D-CNN model:"
for _ in xrange(1):

    history = model_2.fit_generator(class_1d_input.produce_seq_from_2d(get_attributes),nb_example_to_train,epochs,verbose=1)
    print model_2.metrics_names
    model_2.save('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/1D-CNN/1dfight32.h5')
