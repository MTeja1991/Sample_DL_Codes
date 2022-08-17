# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:17:33 2018

@author: mteja
"""
    
# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
#K.set_image_dim_ordering('th')    #for keras 2.2.4 below
K.common.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import BatchNormalization,Input
from keras.optimizers import SGD,RMSprop,adam

#%%

PATH = os.getcwd()
# Define data path
data_path = PATH + '/data1'
data_dir_list = os.listdir(data_path)
for item in data_dir_list:
    if item.endswith(".ini"):
        os.remove(os.path.join(data_path, item))

img_rows=256
img_cols=256
num_channel=3
num_epoch=200

# Define the number of classes
img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    for item in img_list:
        if item.endswith(".ini"):
            os.remove(os.path.join(data_path+'/'+ dataset, item))
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)
#
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
        


#%% test images separately      
PATH = os.getcwd()
# Define data path
data_path1 = PATH + '/New folder'
data_dir_list1 = os.listdir(data_path1)
for item in data_dir_list1:
    if item.endswith(".ini"):
        os.remove(os.path.join(data_path1, item))
test_data_list=[]

for dataset in data_dir_list1:
    img_list=os.listdir(data_path1+'/'+ dataset)
    for item in img_list:
        if item.endswith(".ini"):
            os.remove(os.path.join(data_path1+'/'+ dataset, item))
    img_list=os.listdir(data_path1+'/'+ dataset)
    print ('Loaded the images of test dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path1 + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
        test_data_list.append(input_img_resize)

test_data1 = np.array(test_data_list)
test_data1 = test_data1.astype('float32')
test_data1 /= 255
print (test_data1.shape)
#
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		test_data1= np.expand_dims(test_data1, axis=1) 
		print (test_data1.shape)
	else:
		test_data1= np.expand_dims(test_data1, axis=4) 
		print (test_data1.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		test_data1=np.rollaxis(test_data1,3,1)
		print (test_data1.shape)

#%%
# Assigning Labels
num_of_samples = img_data.shape[0]
num_of_samples1= test_data1.shape[0]

#names = ['AnsweringPhoneCall', 'Archery', 'BasketBall', 'BikeKicking', 'Boating', 'Bowling', 'BowlingThrow', 'Boxing', 'BreakingStick', 'BrushingTeeth', 'CatchingBall', 'Clapping', 'CleaningUtensils', 'CobwebCleaning', 'CombingHair', 'Coughing', 'CountingMoney', 'CricketBating', 'Crouching', 'CuttingMeet', 'CuttingVegitables', 'CuttingWoodAxe', 'CuttingWoodSaw', 'DoubleHandWaving', 'DrinkingTea', 'DrinkingWater', 'Eating', 'ForwardBending', 'GolfSwingPitchShot', 'GolfSwingShortShot', 'Grabbing', 'HammeringNail', 'HammeringRock', 'Hockey', 'ImageCapture', 'IroningClothes', 'JavelinThrow', 'JerkinZipping', 'Jogging', 'KickingSoccerBall', 'KneelDown', 'KnockingDoor', 'LeftBending', 'LeftHandWaving', 'LeftKick', 'LeftPunch', 'Lifting', 'MakingTeaIndianStyle', 'Marchpast', 'Mopping', 'NailBiting', 'OpeningDoor', 'OpeningDrinkBottle', 'OpeningWaterBottle', 'PaintWithRoller', 'PickingWaterFromWell', 'PlayingDrum', 'PlayingFlute', 'PlayingGuitar', 'PlayingHarmonium', 'PlayingJazz', 'PlayingMouthorgan', 'PlayingMridingam', 'PlayingPiano', 'PlayingTabla', 'PlayingTrumpet', 'PlayingVeena', 'PlayingViolin', 'PullingDoor', 'PullingThread', 'Pushing', 'Reading', 'RightBending', 'RightKick', 'RightPunch', 'RotatingHead', 'RubbingHands', 'Running', 'Salutation', 'ScrewTightining', 'Sewing', 'Shooting', 'ShufflingCards', 'Sneezing', 'SowingSeeds', 'StaggerWalk', 'SteeringCar', 'Sweeping', 'TearingPaper', 'TieShoeLace', 'TipToe', 'Typing', 'VideoCapture', 'VolleyBall', 'Walking', 'WalkingTurnLeft90', 'WalkingTurnRight90', 'WallPainting', 'WashingClothes', 'WateringPlants', 'WearingBlazer', 'WearingHat']

names=['answering_a_phone_call-sub-1', 'archery-sub-1', 'basketball-sub-1', 'bike_kick-starting-sub-1', 'boating-sub-1', 'bowling-sub-1', 'bowling_throw-sub-1', 'boxing-sub-1', 'breaking_stick-sub-1', 'brushing_teeth-sub-1', 'catching_ball-sub-1', 'clapping-sub-1', 'cleaning_utensils-sub-1', 'cobweb_cleaning-sub-1', 'combing_hair-sub-1', 'coughing-sub-1', 'counting_money-sub-1', 'cricket_bating-sub-1', 'crouching-sub-1', 'cutting_meat-sub-1', 'cutting_vegetables-sub-1', 'cutting_wood(axe)-sub-1', 'cutting_wood(saw)-sub-1', 'double_hand-waving-sub-1', 'drinking_tea-sub-1', 'drinking_water-sub-1', 'eating-sub-1', 'forward_bending-sub-1', 'golf_swing_pitch_shot-sub-1', 'golf_swing_short_shot-sub-1', 'grabbing-sub-1', 'hammering_nail-sub-1', 'hammering_rock-sub-1', 'hockey-sub-1', 'image_capture-sub-1', 'ironing_clothes-sub-1', 'javelin_throw-sub-1', 'jerkin_zipping-sub-1', 'jogging-sub-1', 'kicking_soccer_ball-sub-1', 'kneel_down-sub-1', 'knocking_door-sub-1', 'left_bending-sub-1', 'left_hand-waving-sub-1', 'left_kick-sub-1', 'left_punch-sub-1', 'lifting-sub-1', 'making_tea_Indian_style-sub-1', 'march-past-sub-1', 'mopping-sub-1', 'nail_biting-sub-1', 'opening_door-sub-1', 'opening_drink_bottle-sub-1', 'opening_water_bottle-sub-1', 'painting(roller)-sub-1', 'picking_water_from_well-sub-1', 'playing_drum-sub-1', 'playing_flute-sub-1', 'playing_guitar-sub-1', 'playing_harmonium-sub-1', 'playing_jazz-sub-1', 'playing_mouth-organ-sub-1', 'playing_mridangam-sub-1', 'playing_piano-sub-1', 'playing_tabla-sub-1', 'playing_trumpet-sub-1', 'playing_veena-sub-1', 'playing_violin-sub-1', 'pulling_door-sub-1', 'pulling_thread-sub-1', 'pushing-sub-1', 'Put_on_hat-sub-1', 'reading-sub-1', 'right_bending-sub-1', 'right_kick-sub-1', 'right_punch-sub-1', 'rotating_head-sub-1', 'rubbing_hands-sub-1', 'running-sub-1', 'salutation-sub-1', 'screw_tightening-sub-1', 'sewing-sub-1', 'shooting-sub-1', 'shuffling_cards-sub-1', 'sneezing-sub-1', 'sowing_seeds-sub-1', 'stagger_walk-sub-1', 'steering_a_car-sub-1', 'sweeping-sub-1', 'tearing_a_paper-sub-1', 'tie_shoelace-sub-1', 'tiptoe-sub-1', 'typing-sub-1', 'video_capture-sub-1', 'volleyball-sub-1', 'walk-turn-left_90-sub-1', 'walking-sub-1', 'walking-turn-right_90-sub-1', 'wall_painting-sub-1', 'washing_clothes-sub-1', 'watering_plants-sub-1', 'wearing_blazer-sub-1']
num_classes = len(names)
#names = ['Kati_LHD', 'Kati_LHU', 'Kati_LHU-RightBend', 'Kati_RHD', 'Kati_RHU', 'Kati_RHU-LeftBend', 'Thad_HandsDown', 'Thad_HandsUp', 'Thad_Tad', 'TThad_HandsDown_LegStretch', 'TThad_HandsUp-LegStretch', 'TThad_Legs2Normal', 'TThad_LegStretch',' TThad_TTad', 'TThad_TTad_Left', 'TThad_TTad_Right', 'Trik_HandsDown', 'Trik_HandsUp', 'Trik_LeftBend', 'Trik_LegsNormal', 'Trik_LegStretch', 'Trik_RightBend', 'Vruk_HandsDown', 'Vruk_HandsUp', 'Vruk_RLegFold', 'Vruk_RLegUnfold']
labels = np.ones((num_of_samples,),dtype='int64')

labels1 = np.ones((num_of_samples1,),dtype='int64')
j=0
k=0
for i in names:
    labels[j:]=k
    j+=15
    k+=1

j=0
k=0
for i in names:
    labels1[j:]=k
    j+=5
    k+=1
    

#labels1[0:]=0    
#labels1[243:]=1  
#labels1[387:]=2  
#labels1[675:]=3  
#labels1[927:]=4  
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
Y1 = np_utils.to_categorical(labels1, num_classes)
##Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=1)
#x1,y1 = shuffle(test_data1,Y1, random_state=1)

X_train = img_data
y_train = Y
X_test = test_data1
y_test = Y1
### Split the dataset
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.07, random_state=1)

#%%
import math
from keras.callbacks import LearningRateScheduler
 
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.1
    epochs_drop = 40.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #lrate = initial_lrate+0.05
    return lrate

#%%
X_input = Input(shape=img_data[0].shape)

#Low level feature
X = Conv2D(16, (3, 3) ,strides=(2, 2) , name = 'conv01',kernel_initializer = glorot_uniform(seed=0))(X_input)
X = BatchNormalization( name = 'bn_conv01')(X)
X = Activation('relu')(X)

X = Conv2D(32, (5, 5) ,strides=(2, 2), name = 'conv02', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(name = 'bn_conv02')(X)
X = Activation('relu')(X)

X = Conv2D(64, (7, 7) ,strides=(2, 2), name = 'conv03', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization( name = 'bn_conv03')(X)
X = Activation('relu')(X)


X = Conv2D(128, (9, 9) ,strides=(2, 2) , name = 'conv04', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(name = 'bn_conv04')(X)
X = Activation('relu')(X)
X = keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)(X)
X = MaxPooling2D((2, 2))(X)


# Mid level feature
Y = Conv2D(16, (5, 5) ,strides=(2, 2),name = 'conv11', kernel_initializer = glorot_uniform(seed=0))(X_input)
Y = BatchNormalization(name = 'bn_conv11')(Y)
Y = Activation('relu')(Y)


Y = Conv2D(32, (7, 7) ,strides=(2, 2), name = 'conv12', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization(name = 'bn_conv12')(Y)
Y = Activation('relu')(Y)


Y = Conv2D(64, (9, 9) ,strides=(2, 2), name = 'conv13', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization(name = 'bn_conv13')(Y)
Y = Activation('relu')(Y)


Y = Conv2D(128, (3, 3) ,strides=(2, 2), name = 'conv15', kernel_initializer = glorot_uniform(seed=0))(Y)
Y = BatchNormalization( name = 'bn_conv15')(Y)
Y = Activation('relu')(Y)
Y = MaxPooling2D((2, 2))(Y)

# Mid level feature
Z = Conv2D(16, (7, 7) ,strides=(2, 2), name = 'conv21', kernel_initializer = glorot_uniform(seed=0))(X_input)
Z = BatchNormalization( name = 'bn_conv21')(Z)
Z = Activation('relu')(Z)


Z = Conv2D(32, (9, 9) ,strides=(2, 2), name = 'conv22', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization(name = 'bn_conv22')(Z)
Z = Activation('relu')(Z)


Z = Conv2D(64, (3, 3) ,strides=(2, 2), name = 'conv23', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization( name = 'bn_conv23')(Z)
Z = Activation('relu')(Z)


Z = Conv2D(128, (5, 5) ,strides=(2, 2), name = 'conv24', kernel_initializer = glorot_uniform(seed=0))(Z)
Z = BatchNormalization( name = 'bn_conv24')(Z)
Z = Activation('relu')(Z)
Z = MaxPooling2D((2, 2))(Z)

# High level feature
Z1 = Conv2D(16, (9, 9) , strides=(2, 2),name = 'conv31', kernel_initializer = glorot_uniform(seed=0))(X_input)
Z1 = BatchNormalization( name = 'bn_conv31')(Z1)
Z1 = Activation('relu')(Z1)


Z1 = Conv2D(32, (3, 3) ,strides=(2, 2), name = 'conv32', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv32')(Z1)
Z1 = Activation('relu')(Z1)

Z1 = Conv2D(64, (5, 5) ,strides=(2, 2), name = 'conv33', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv33')(Z1)
Z1 = Activation('relu')(Z1)


Z1 = Conv2D(128, (7, 7) ,strides=(2, 2), name = 'conv34', kernel_initializer = glorot_uniform(seed=0))(Z1)
Z1 = BatchNormalization( name = 'bn_conv34')(Z1)
Z1 = Activation('relu')(Z1)
Z1 = MaxPooling2D((2, 2))(Z1)

add=keras.layers.concatenate([X,Y,Z,Z1],axis=1)
#drp=Dropout(0.2)(add)
#conv=Conv2D9(256,(7,7),kernel_initializer = glorot_uniform(seed=0))(add)
#relu=Activation('relu')(conv)
#Apool = AveragePooling2D(pool_size=(3, 3), padding='valid')(relu)
#
#flt = Flatten()(Apool)
flt = Flatten()(add)

dense = Dense(102, activation='softmax')(flt)
# Stage 2
# Create model
model = Model(inputs = X_input, outputs = dense)

sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=["accuracy"])
#model.compile(loss='mean_squared_error', optimizer='sgd',metrics=["accuracy"])


model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
#a=model.layers[0].get_weights()
#np.shape(model.layers[0].get_weights()[0])
#model.layers[0].trainable
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#%%
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate] 
# Training
hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, callbacks=callbacks_list, verbose=1, validation_data=(X_test, y_test))

#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=20,verbose=1, validation_split=0.2)

## Training with callbacks
#from keras import callbacks
#
#filename='model_train_new.csv'
#csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
#
#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
#
#filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
#
#checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#
#callbacks_list = [csv_log,early_stopping,checkpoint]
#
#hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))


# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(8,6))
plt.plot(xc,train_loss,'blue')
plt.plot(xc,val_loss,'orange')
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid()
plt.legend(['train','val'],loc=1)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])
plt.savefig('loss plot')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc,'blue')
plt.plot(xc,val_acc,'orange')
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid()
plt.legend(['train','val'],loc=1)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['seaborn-white'])
plt.savefig('accuracy plot')
#%%
#model.layers[0].get_config()
#
#
#model.get_config()
#
#model.layers[0].count_params()
#
#model.count_params()
#
#model.layers[0].kernel
#
#model.layers[0].kernel.get_shape()
#
##%%
# visualizing kernal data in convolution layer
#import keras
#i=2
#def plot_filters(layers,x,y):
#    filters=keras.backend.get_value(model.layers[i].filters)
#    filters=np.rollaxis(filters,3,0)
#    filters=np.rollaxis(filters,3,1)
#    fig=plt.figure(figsize=(16,16))
#    for j in range(len(filters)):
#        ax=fig.add_subplot(y,x,j+1)
#        ax.matshow(filters[j][0],cmap='brg')
#        plt.xticks(np.array([]))
#        plt.yticks(np.array([]))
#    plt.tight_layout()
#    return plt
##
#plot_filters(model.layers[i],4,4)

#%%
# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
#print(model.predict_classes(test_image))
print(y_test[0:1])

# Testing a new image
test_image = cv2.imread('D:/Volume-MultiCNN/Volume-Train/CricketBating/Sub1_Cricket_Bating75_YZ_Volume.jpg')
#test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(img_rows,img_cols))
test_image = np.array(test_image )
test_image = test_image.astype('float32')
test_image /= 255
print (test_image .shape)
   
if num_channel==1:
	if K.common.image_dim_ordering=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.common.image_dim_ordering=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test image
print((model.predict(test_image)))
#print(model.predict_classes(test_image))

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

for x in range(1,55):
    layer_num=x
    filter_num=0
    
    activations = get_featuremaps(model, int(layer_num),test_image)
    
    print (np.shape(activations))
    feature_maps = activations[0][0]      
    print (np.shape(feature_maps))
    
    if K.common.image_dim_ordering=='th':
    	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
    print (feature_maps.shape)
    
    fig=plt.figure(figsize=(16,16))
    plt.imshow(feature_maps[:,:,filter_num],cmap='jet')
    fig.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.png')
    
    num_of_featuremaps=feature_maps.shape[2]
    fig=plt.figure(figsize=(16,16))	
    plt.title("featuremaps-layer-{}".format(layer_num))
    subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
    for i in range(int(num_of_featuremaps)):
    	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    	ax.imshow(feature_maps[:,:,i],cmap='jet')
    	plt.xticks([])
    	plt.yticks([])
    	plt.tight_layout()
    plt.show()
    fig.savefig("featuremaps-layer-{}".format(layer_num) + '.png')


#%%
#Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred,axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pr/ed)
target_names = names
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)
fig=plt.figure(figsize=(15,15))
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.show()
fig.savefig('cnf')

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')
#%%
C=names
List=[]
j=0
for i in test_data_list :
    c=y_pred[j]
    l=C[c]
    List.append(l);
    j+=1
print(List)    


#%%
#L=List
#import pyttsx3
#engine  = pyttsx3.init()
#voices = engine.getProperty('voices')
#engine.setProperty('voice', voices[1].id) #change index to change voices
#engine.say(L)
#engine.runAndWait()
