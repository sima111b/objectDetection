# This code provides a simple object detector using VGGNet
# cv2 is required for selective search
# Please run "pip install opencv-contrib-python" to use selective search
#importing libraries
import os,keras, cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import calc_iou
import experimentReport
import extractImages
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from dataSplit import MyLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
#the directory of the images
dirpath = "Images"
#the object classes anootations
annotations = "Airplanes_Annotations"
trainSet,trainLabel=extractImages(annotations,dirpath)
train_images = np.array(trainSet)
train_labels = np.array(trainLabel)
# ===================================== Preparing the fine-tunning model ===============================================
vggmodel = VGG16(weights='imagenet', include_top=True)
for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False
X= vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)
opt = Adam(lr=0.0001) #set the learning rate for Adam optimization
model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_final.summary()
#one-hot encode
encoder = MyLabelBinarizer()
Y =  encoder.fit_transform(train_labels)
#========================================= Data Augmentation ===========================================================
X_train, X_test , y_train, y_test = train_test_split(train_images,Y,test_size=0.10)
# Image augmentation to increase the dataset size
trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)
#========================================= Train phase =================================================================
checkpoint = ModelCheckpoint("objDtr_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, \
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
summery = model_final.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 1000, validation_data= testdata, \
                                  validation_steps=2, callbacks=[checkpoint,early])
experimentReport(summery)
#============================================ Test phase ===============================================================
test_samples=0
ssearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() # initializing selective search
for entries in (os.listdir(dirpath)):
    if entries.startswith("4"): # the test samlples names starts with 4
        test_samples += 1
        img = cv2.imread(os.path.join(dirpath,entries))
        ssearch.setBaseImage(img)
        ssearch.switchToSelectiveSearchFast()
        boxProposals = ssearch.process()
        image_copy = img.copy()
        for ctr,bxp in enumerate(boxProposals):
            if ctr < 2000:
                x,y,w,h = bxp
                croppedImage = image_copy[y:y+h,x:x+w]
                resized = cv2.resize(croppedImage, (224,224), interpolation = cv2.INTER_AREA)
                tst_img = np.expand_dims(resized, axis=0)
                out= model_final.predict(tst_img)
                if out[0][0] > 0.65:
                    cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image_copy)
