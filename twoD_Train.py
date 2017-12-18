from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.models import load_model
from Functions_2 import myGenerator
    # if yield_bool == True:
    #     while 1:
    #       for i in range(len(X_train)- seq_size):
    #
    #         yield X_train[i :(i + seq_size)],Y_train[i:(i + seq_size)]
#
#
# import pickle


# "randomize"
# data = []
# target = []
# rand = np.arange(0,(len(data_)))
# np.random.shuffle(rand)
# print" Shuffled list :", rand
# for _ in rand :
#     data.append(data_[_])
#     target.append(target_[_])
# data = np.array(data_)
# target = np.array(target_)
# print "After randomizing data =  ", len(data),"label = ",len(target)
# with open('data_youtube.pickle', 'wb') as output:
#     pickle.dump(data_, output)
# with open('Label_youtube.pickle', 'wb') as output:
#         pickle.dump(target_, output)
# with open('data_youtube.pickle', 'rb') as data:
#     data = pickle.load(data)
# with open('Label_youtube.pickle', 'rb') as target:
#     target = pickle.load(target)
# model = load_model('.h5')
Path = 'C:\Users\USER\Desktop\Downloaded DataSets\Weizmann_train'
batch_size = 32
nb_classes = 10
nb_epoch = 10
red_size = 64
nb_examples = 4599
data,target = myGenerator(Path,red_size)
def twod_net():
    "This is the 2d network...."

    inputs_2d = Input(shape=(red_size,red_size,1))

    conv_1 = Convolution2D(2,3,3,activation='sigmoid',border_mode='same')(inputs_2d)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = Convolution2D(4,3,3, activation='relu', border_mode='same')(pool1)
    max2 = MaxPooling2D(pool_size=(2,2))(conv_2)

    conv_3 = Convolution2D(8,3,3,activation='relu',border_mode='same')(max2)
    max3 = MaxPooling2D(pool_size=(2,2))(conv_3)

    conv_4 = Convolution2D(8,3,3,activation='relu',border_mode='same')(max3)
    max4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    conv_5 = Convolution2D(16,3,3,activation='relu',border_mode='same')(max4)
    max5 = MaxPooling2D(pool_size=(2,2))(conv_5)



    flat = Flatten()(max5)
    # dense_1 = Dense(12,activation='relu')(flat)
    dense_2= Dense(nb_classes,activation='softmax')(flat)

    model = Model(input=inputs_2d, output=dense_2,name='model_1')
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    print "Summary of the Model:", (model.summary())

    return model

# model = twod_net()
model = load_model("weizmann_train_64_new.h5")
print model.summary()
# print 'reaching ...'
# model.fit_generator(myGenerator(path,10),samples_per_epoch=5701,nb_epoch=2,verbose=1)
for _ in xrange(50):
    print "Entering Epoch", _ + 1
    model.fit(data,target,batch_size,1)

    model.save('weizmann_train_64.h5')

