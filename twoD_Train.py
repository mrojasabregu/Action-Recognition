from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model
from Functions_2 import myGenerator_optical
import pickle

"randomize video data if necessary"
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
"Data set dirctory"
Path = 'C:\Users\USER\Desktop\Fight data set'
batch_size = 32; "Batch size for the Convolutional Neural Ntwork"
nb_classes = 2;"Number of classes"
nb_epoch = 10;"Number of Iterations for Backpropagation"
red_size = 32;"Resize the video (N x N x 3 x t)"
# nb_examples = 4599

data,target = myGenerator_optical(Path,red_size)
"The function returns the processed extracted frames and labels from the directory. The processing involves application of Optical flow " \
"algorithm and color coding the resultant vectors representing the magnitude and direction of Optical flow. This allows to impart " \
"some temporal information in the frames extracted, which could be classified. The resizing is also ensured, given the computational " \
"complexity associated with the videos."

"Load processed files if already saved"
with open('fightdata.pickle', 'wb') as output:
      pickle.dump(data,output)
with open('fightlabel.pickle', 'wb') as output:
      pickle.dump(target,output)


"Custom defined two dimensional neural network for extracting the spatial information from the processed frames. Instad of " \
"using pretrained models, the dataset is trained from scratch to ensure that the model is highly specific to the task and" \
"computational complexity could b easily managed "
def twod_net(optical):
    if optical == True:
      inputs_2d = Input(shape=(red_size,red_size,3))
    if optical == False:
      inputs_2d = Input(shape=(red_size, red_size, 1))
    " Depending on whether Optical flow or simple background subtraction is used in the extracted frames"

    "Two dimensional Convolutional neural network architecture:"

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
    "Compilation:"
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
    print "Summary of the Model:", (model.summary())

    return model

# model = twod_net(optical=True)
model = load_model("C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/2D-Cnn/opticalfight32.h5")
print model.summary()
for _ in xrange(50):
    print "Entering Epoch", _ + 1
    model.fit(data,target,batch_size,1)
    "Saving the model:"
    model.save('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/2D-Cnn/opticalfight32.h5')

