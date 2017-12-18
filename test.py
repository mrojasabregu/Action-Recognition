from Functions_2 import test
from keras.models import load_model
test_video = 'walk'
video_no = 1
path = 'C:\Users\USER\Desktop\Downloaded DataSets\weizmann_test/' + str(test_video)
# model_2d = load_model('wiezman_backnewblack1.h5')
# model1= load_model('WIEZMAN_back1dnewblack_seq20.h5')
model1 = load_model('C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\models_save\Weizmann_train_64_9812_epoch4_seq10.h5')
model_2d = load_model('weizmann_train_64.h5')
seq_size = 10
Nodes_flatten = 64
reduced_size = 64
layer_no = 11
tesing = test(path,video_no-1,model_2d,model1,seq_size,Nodes_flatten,reduced_size,layer_no,False,False)
tesing.test()
#tesing.render_video()