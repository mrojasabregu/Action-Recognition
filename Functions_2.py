from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from os.path import join,isfile
from keras import backend as k
from keras.utils import np_utils
"generates the train and test data"
def myGenerator_optical(path,reduced_size):
    Y_train = []
    X_train = []
    list = os.listdir(path)

    for i in xrange(len(list)):
        list_dir = glob.glob(path + str('/') + str(list[i]) + str('/*'))
        total_count = 0
        for _ in list_dir:
            cap = cv2.VideoCapture(_)
            # Take the first frame and convert it to gray
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Create the HSV color image
            hsvImg = np.zeros_like(frame)
            hsvImg[..., 1] = 255
            count = 0
            # n = 0
            while True:
                # Save the previous frame data
                previousGray = gray

                # Get the next frame
                ret, frame = cap.read()

                if ret:
                    # Convert the frame to gray scale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate the dense optical flow
                    flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    # Obtain the flow magnitude and direction angle
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # print mag, ang

                    # Update the color image
                    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
                    hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
                    frame = cv2.resize(np.array(rgbImg), (reduced_size, reduced_size))
                    # print frame.shape
                    # cv2.imshow('dense optical flow', frame )
                    # k = cv2.waitKey(30) & 0xff
                    # if k == 27:
                    #     break

                if ret == False:
                    print "Number of Frames Extracted :", count
                    break
                count = count + 1
                X_train.append(frame)

            total_count = total_count + count

        for j in xrange(total_count):
            Y_train.append(i)

        print 'Directory ${0}$ Extracted with total frames : {1}\n'.format(list[i], total_count)

    print ('Sucessfully Extracted {0} frames with {1} labels'.format(len(X_train), len(Y_train)))

    Y_train = np_utils.to_categorical(Y_train, len(list))
    X_train = np.array(X_train).reshape(-1, reduced_size, reduced_size, 3)
    print X_train.shape,Y_train.shape
    # X_train = np.array(X_train)
    # if yield_bool == False:
    return X_train, Y_train

def myGenerator(path,reduced_size):
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
            # n = 0
            while(cap.isOpened()):
             ret, frame = cap.read()
             if(ret == True):
                 # if (n <= 4):
                 #     continue
                 frame =np.array(back_sep.apply(frame))

                 retu , thr = cv2.threshold(frame, 0, 136, cv2.THRESH_BINARY)
                     # print thr.shape
                 frame = cv2.resize(thr,(reduced_size,reduced_size))

                     # print np.array(frame).shape
                     # frame = plt.imread(frame)
                     # plt.imshow(frame)
                     # plt.show()
                 # cv2.imshow('frame',frame)
                 # k = cv2.waitKey(0) & 0xff
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
    X_train = np.array(X_train).reshape(-1,reduced_size,reduced_size,1)
    # X_train = np.array(X_train)
    # if yield_bool == False:
    return X_train, Y_train

" generates the vectors from the two dimensional convolution neural network and compiles the sequences of frames to be fed " \
"to one dimensional neural network to extract temporal features from the frames"
class piece_wise:
    def __init__(self,model,data,label,seq_size,layer_idx,*args,**kwargs):
        self.args = args
        self.count = 1
        # self.batch_size = batch_size
        self.kwargs = kwargs
        self.seq = seq_size
        self.model = model
        self.layer_idx = layer_idx
        self.data = data
        self.label = label
        self.examples = data.shape[0]
        self.i = 0
        # self.j = -1
        # self.count2 = 0
        # self.acti = []
        # self.label2 = []
        self.arr = np.arange(0,(self.examples-self.seq))
        np.random.shuffle(self.arr)
    def gen_seq(self):
        "This function yields for the sequence training with seq prompted"
        # data = self.data
        # label= self.label
        "while was eliminated to see if code runs normally"

        if self.count == self.data.shape[0]-self.seq:
            print "Going into recurrsion..."
            self.i = 0
            self.count = 0
            self.gen_seq()

        j = self.arr[self.i]

        self.i = self.i + 1
        self.count = self.count + 1
        "See if code runs with desirable results with if statement"
        _ = self.data[j:(j+self.seq)]
        __ = self.label[j:(j + self.seq)]
        # _ = data[self.count:(self.count + self.seq)]
        # __ = label[self.count:(self.count + self.seq)]
        # print '|'
        # if _.shape[0] != self.seq:
        #     self.i = 0
        #     self.gen_seq()
        yield _ , __


    def produce_seq_from_2d(self,get_attribute):
        # for _ in xrange((self.examples - self.seq)/self.batch_size):
        #   data = []
        #   label =[]
        # for __ in xrange(self.examples - self.seq):
            # t = time()
        # if self.j == 32:
        #     self.acti = []
        #     self.label2 = []
        #     self.j = -1
        #     self.count2 = 0

        while 1:
            gen = next(self.gen_seq())
            # self.j =  self.j + 1
            # self.count2 = self.count2 + 1
            if np.array_equal(gen[1][0:self.seq/2],gen[1][self.seq/2: self.seq]):
                act = np.array(get_attribute([gen[0]]))

            # t= time()
            # act = np.array(attribute(gen[0]))
            # print act.shape
            # print time()-t
            # print len(gen[0])
                act = np.reshape(act,(-1,self.seq,act.shape[2]))

            # x = load_ped_centrist(gen[0])
            # act = np.hstack((act,x)).reshape(1,self.seq,-1)
            # print act
            # data.append(act)
                la = gen[1][5]
            # label.append(la)
            # print time() - t, "sec"
            #     self.acti.append(act)
            #     self.label2.append(la)
                # if self.j == 0:
                #     self.acti = act
                #     self.label2 = la
                # if self.j > 0:
                #     self.acti = np.append(self.acti,act,axis=0)
                #     print self.acti.shape
                #     self.label2 = np.append(self.label2,la,axis=0)
                # _ = np.array(self.acti)
                # __ = np.array(self.label2).reshape(self.count,self.seq)
                yield act,np.array(la).reshape((1, la.shape[0]))
                # yield _,__
             # print "no"
             # continue
             # break
        # print self.count2
        # print np.array(self.acti).reshape(1,10,64).shape,np.array(self.label2).reshape((self.count2,self.seq)).shape
        #
        # yield np.array(self.act).reshape(1,10,64),np.array(self.la).reshape((self.count2,self.seq))

"returns the metrics for classification and visualization"
class test:
    def __init__(self, path, file_no, model_2d, model_1d, seq_size,Nodes_flatten,reduced_size,optical,consec,layer_no,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs
        self.path = path
        self.optical = optical
        self.layer_no = layer_no
        self.r_size = reduced_size
        self.consec = consec
        # self.test_name = test_name
        self.file_no = file_no
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.seq_size = seq_size
        self.Nodes_flatten = Nodes_flatten
        self.get_activations = k.function([self.model_2d.layers[0].input,k.learning_phase()], self.model_2d.layers[layer_no].output)
        self.preds = []
        # self.list = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']
        self.list = ['Fight', 'Normal']
        # self.list = ['basketball','biking','diving','golf_swing','horse_riding','soccer_juggling','swing','tennis_swing','trampoline_jumping','voleyball_spiking','walking']
    def test_gen(self):
        file = [join(self.path, f) for f in os.listdir(self.path) if isfile(join(self.path, f))][self.file_no]
        cap = cv2.VideoCapture(file)
        # cap = cv2.VideoCapture(0)
        data = []
        count = 0
        if self.optical == True:
            y = 3
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsvImg = np.zeros_like(frame)
            hsvImg[..., 1] = 255
            while True:
                previousGray = gray
                ret, frame = cap.read()

                if ret:
                    # Convert the frame to gray scale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate the dense optical flow
                    flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    # Obtain the flow magnitude and direction angle
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # print mag, ang

                    # Update the color image
                    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
                    hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
                    dump = cv2.resize(rgbImg, (self.r_size, self.r_size))
                    # cv2.imshow('dense optical flow', rgbImg)
                    # if cv2.waitKey(30) & 0xFF == ord('q'):
                    #     break
                    # print np.array(dump).shape
                    count = count + 1
                    # print count
                    data.append(dump)
                    r = count
                if ret == False:
                        if self.consec == False:
                            appen = []
                            mod = len(data) % self.seq_size
                            if mod >= self.seq_size / 2:
                                for _ in xrange(self.seq_size - mod):
                                    appended = [0 for i in xrange(self.r_size * self.r_size*3)]
                                    appended = np.array(appended).reshape(self.r_size, self.r_size,3)
                                    appen.append(appended)
                                    #     # print "hoooooo"
                                    #     print np.array(appen).shape
                                data = np.vstack((np.array(data), np.array(appen)))
                                print "Empty frames added:", self.seq_size - mod

                                r = count + self.seq_size - mod
                                break
                            data = data[0:len(data) - mod]
                            r = count - mod
                            print "Frames removed:", mod
                            break
                        break
        if self.optical == False:
             y = 1
             back_sep = cv2.createBackgroundSubtractorKNN()
             while (cap.isOpened()):
                    ret, frame = cap.read()

                    if (ret == True):
                        odd = frame.shape
                        # mask = np.zeros(np.array(frame).shape[:2],np.uint8)
                        # bgd = np.zeros((1,65),np.float64)
                        # fwd = np.zeros((1,65),np.float64)
                        # rect = (20,20,50,50)
                        # cv2.grabCut(frame,mask,rect,bgd,fwd,5,cv2.GC_INIT_WITH_RECT)
                        # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                        # frame= frame*mask2[:,:,np.newaxis]
                        # plt.imshow(frame),plt.colorbar(),plt.show()
                        # dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
                        dump = back_sep.apply(frame)
                        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                        # print frame.shape
                        retu, thr = cv2.threshold(dump, 0 , 136 , cv2.THRESH_BINARY)
                        dump = cv2.resize(thr, (self.r_size, self.r_size))
                        # print np.array(dump).shape
                        count = count + 1
                        # cv2.imshow('frame',dump)
                        # k = cv2.waitKey(30) & 0xff
                        # if k ==27:
                        #  break
                        data.append(dump)
                        r = count
                    if ret == False:
                        if self.consec == False:
                            appen = []
                            mod = len(data) % self.seq_size
                            if mod >= self.seq_size/2:
                              for _ in xrange(self.seq_size-mod):
                                appended = [0 for i in xrange(self.r_size * self.r_size)]
                                appended = np.array(appended).reshape(self.r_size,self.r_size)
                                appen.append(appended)
                            #     # print "hoooooo"
                            #     print np.array(appen).shape
                              data = np.vstack((np.array(data), np.array(appen)))
                              print "Empty frames added:", self.seq_size-mod

                              r = count + self.seq_size-mod
                              break
                            data = data[0:len(data)-mod]
                            r = count - mod
                            print "Frames removed:", mod
                            break
                        break
        return np.array(data).reshape(-1,self.r_size,self.r_size,y), file, r
        # return np.array(data), file
    def test(self):
        data, file, count1 = self.test_gen()
        print "Video shape after operation:", data.shape
        i = -1
        timer = []
        cap = cv2.VideoCapture(file)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                 break
            i = i + 1
            if i == count1-self.seq_size+1:
               break
            start = time()
            if self.consec == False:
              act = self.get_activations([data[i*self.seq_size:(i + 1)* self.seq_size],0]).reshape(1, self.seq_size, self.Nodes_flatten)
            if self.consec == True:
              act = self.get_activations([data[i:i + self.seq_size], 0]).reshape(1, self.seq_size, self.Nodes_flatten)
            # print np.array(act).shape
            # if act.shape != (self.seq_size,self.Nodes_flatten):
            #
            #     break
            # act = np.reshape(act,(1, self.seq_size, self.Nodes_flatten))
            pre = self.model_1d.predict(act)
            at = np.argmax(pre)
            ac = np.max(act)
            stop = time() - start
            timer.append(stop)
            self.preds.append(at)
            if self.consec == True:
                "Remove this for saving part~~~~"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(self.list[at] + " " + str(ac)), (70, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Prediction', frame)
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

                "Remove this for saving part~~~~"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
        cap.release()
        cv2.destroyAllWindows()
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        Unique, count = np.unique(np.array(self.preds), return_counts=True)
        print Unique,count
        j = 0
        den = np.sum(count)
        tim = (float(np.sum(timer)) / den).round(2)
        print "Total Frames Extracted:", den
        for _ in Unique:
            print "Predicted:", '$', self.list[_], '$', "with count:", count[j], "Accuracy:", (
                (float(count[j]) / den) * 100).round(2), "%"
            j = j + 1
        print "Average time taken per frame:", tim, "sec"
        print "Average frame rate:", (float(1) / tim).round(2)
        return self.preds, Unique[np.argmax(count)],file, Unique, count, timer,ac

    def render_video(self):

     pred, argmax, file, Unique, count, timer,ac = self.test()
     j = 0
     den = np.sum(count)
     tim = (float(np.sum(timer)) / den).round(2)
     print "Total Frames Extracted:", den
     for _ in Unique:
         print "Predicted:", '$', self.list[_], '$', "with count:", count[j], "Accuracy:", (
             (float(count[j]) / den) * 100).round(2), "%"
         j = j + 1
     print "Average time taken per frame:", tim, "sec"
     print "Average frame rate:", (float(1) / tim).round(2)

     _msg = raw_input("Do You Want To View And Save The Video? Y/N? ")
     if _msg == "Y" or "y":
         _msg2 = raw_input(" Name The File (.avi): ")
         print "Rendering video with maximum predicted class... "
         cap = cv2.VideoCapture(file)
         out = cv2.VideoWriter("C:\Users\USER\PycharmProjects\iisc-hello\SU_Project\Results_appendingframes"+ str('/') +str(_msg2), -1, 20.0, (180, 144))
         while (cap.isOpened()):
             ret,frame = cap.read()
             if ret == True:
                 font = cv2.FONT_HERSHEY_SIMPLEX
                 cv2.putText(frame, str(self.list[argmax] + " " + str(ac)), (70,140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                 out.write(frame)
                 cv2.imshow('Prediction' , frame)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
             if ret == False:
                 break

         cap.release()
         cv2.destroyAllWindows()


        #     for _ in xrange(self.seq_size):
        #         ret, frame = cap.read()
        #         if ret == False:
        #             break
        #         if (ret == True):
        #             font = cv2.FONT_HERSHEY_SIMPLEX
        #             cv2.putText(frame, str(self.list[at]), (70,140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #             out.write(frame)
        #             cv2.imshow('Prediction' , frame)
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 break
        #
        # cap.release()
        # cv2.destroyAllWindows()

