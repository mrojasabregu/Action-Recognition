from Functions_2 import test
from keras.models import load_model
test_video = 'Fight'
video_no = 3
optical = True
consec = True
"False code to be seen"
# path = 'C:\Users\USER\Desktop\Downloaded DataSets\weizmann_test/' + str(test_video)
"Load test Data: "
path = 'C:\Users\USER\Desktop\Fight data set/'+ str(test_video)
"Load 1D-CNN weights:"
model1 = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/1D-CNN/1dfight32.h5')
"Load 2D-CNN weights:"
model_2d = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Optical_results/2D-Cnn\opticalfight32.h5')
seq_size = 20
Nodes_flatten = 16
reduced_size = 32
layer_no = 11

"Test the frames on the saved models:"
tesing = test(path,video_no-1,model_2d,model1,seq_size,Nodes_flatten,reduced_size,optical,consec,layer_no,False,False)
tesing.test()
"for saving:"
# tesing.render_video()