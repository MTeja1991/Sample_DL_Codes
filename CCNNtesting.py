# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:02:13 2018

@author: mteja
"""

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import model_from_json


num_channel=3
img_rows=200
img_cols=200


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
model=loaded_model
PATH = os.getcwd()

#%%
# Define data path
data_path1 = PATH + '/Test_Mod_Vol'
data_dir_list = os.listdir(data_path1)
test_data_list=[]
original=[]
for dataset in data_dir_list:
    	img_list=os.listdir(data_path1+'/'+ dataset)
    	print ('Loaded the images of test dataset-'+'{}\n'.format(dataset))
    	for img in img_list:
            input_img=cv2.imread(data_path1 + '/'+ dataset + '/'+ img )
            original.append(input_img)
            input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
            test_data_list.append(input_img_resize)
            
test_data1 = np.array(test_data_list)
test_data1 = test_data1.astype('float32')
test_data1 /= 255
print (test_data1.shape)
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_data1= np.expand_dims(test_data1, axis=1) 
		print (test_data1.shape)
	else:
		test_data1= np.expand_dims(test_data1, axis=4) 
		print (test_data1.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_data1=np.rollaxis(test_data1,3,1)
		print (test_data1.shape)
 
        
#%% labels
#C=['AnsweringPhoneCall', 'Archery', 'BasketBall', 'BikeKicking', 'Boating', 'Bowling', 'BowlingThrow', 'Boxing', 'BreakingStick', 'BrushingTeeth', 'CatchingBall', 'Clapping', 'CleaningUtensils', 'CobwebCleaning', 'CombingHair', 'Coughing', 'CountingMoney', 'CricketBating', 'Crouching', 'CuttingMeet', 'CuttingVegitables', 'CuttingWoodAxe', 'CuttingWoodSaw', 'DoubleHandWaving', 'DrinkingTea', 'DrinkingWater', 'Eating', 'ForwardBending', 'GolfSwingPitchShot', 'GolfSwingShortShot', 'Grabbing', 'HammeringNail', 'HammeringRock', 'Hockey', 'ImageCapture', 'IroningClothes', 'JavelinThrow', 'JerkinZipping', 'Jogging', 'KickingSoccerBall', 'KneelDown', 'KnockingDoor', 'LeftBending', 'LeftHandWaving', 'LeftKick', 'LeftPunch', 'Lifting', 'MakingTeaIndianStyle', 'Marchpast', 'Mopping', 'NailBiting', 'OpeningDoor', 'OpeningDrinkBottle', 'OpeningWaterBottle', 'PaintWithRoller', 'PickingWaterFromWell', 'PlayingDrum', 'PlayingFlute', 'PlayingGuitar', 'PlayingHarmonium', 'PlayingJazz', 'PlayingMouthorgan', 'PlayingMridingam', 'PlayingTabla', 'PlayingTrumpet', 'PlayingVeena', 'PlayingViolin', 'Playing_Piano', 'PullingDoor', 'PullingThread', 'Pushing', 'Reading', 'RightBending', 'RightKick', 'RightPunch', 'RotatingHead', 'RubbingHands', 'Running', 'Salutation', 'ScrewTightining', 'Sewing', 'Shooting', 'ShufflingCards', 'Sneezing', 'SowingSeeds', 'StaggerWalk', 'SteeringCar', 'Sweeping', 'TearingPaper', 'TieShoeLace', 'TipToe', 'Typing', 'VideoCapture', 'VolleyBall', 'Walking', 'WalkingTurnLeft90', 'WalkingTurnRight90', 'WallPainting', 'WashingClothes', 'WateringPlants', 'WearingBlazer', 'WearingHat']
C=['AnsweringPhoneCall', 'Archery', 'BowlingThrow', 'BreakingStick', 'CleaningUtensils', 'CobwebCleaning', 'CombingHair', 'CountingMoney', 'DrinkingWater', 'ForwardBending', 'GolfSwingPitchShot', 'HammeringNail', 'IroningClothes', 'JavelinThrow', 'Jogging', 'KnockingDoor', 'LeftKick', 'Lifting', 'MakingTeaIndianStyle', 'Mopping', 'OpeningWaterBottle', 'PickingWaterFromWell', 'PlayingGuitar', 'PlayingMridingam', 'PlayingTabla', 'PushingDoor', 'RightKick', 'RubbingHands', 'Sewing', 'Walking']
num_classes = len(C)
num_of_samples1= test_data1.shape[0]

labels1 = np.ones((num_of_samples1,),dtype='int64')
#
j=0
k=0
for i in C:
    labels1[k:]=j
    k+=28
    j+=1
 
Y1 = np_utils.to_categorical(labels1, num_classes)
#x1,y1 = shuffle(test_data1,Y1, random_state=1)
X_test1 = test_data1

y_test = Y1

Y_pred = model.predict(X_test1)
print(Y_pred)
y_pred = np.argmax(Y_pred,axis=1)
print(y_pred)


from sklearn.metrics import classification_report,confusion_matrix
import itertools 

target_names = C
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))


print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
f=30
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
    plt.title(title,fontsize=25)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,horizontalalignment="right", rotation=45,fontsize=f)
    plt.yticks(tick_marks, classes,fontsize=f)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",verticalalignment="center",fontsize=20,
                 color="white" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=f)
    plt.xlabel('Predicted label',fontsize=f)

if (len(test_data1)/len(C)!=1):
    w=int(len(test_data1)/len(C))
    Y1_pred=np.zeros(shape=(num_classes,num_classes))
    j=0
    m=0
    for k in range(len(C)):
        print(j,m,k)
        for i in range(w):
            Y1_pred[k,:]+=Y_pred[j,:]
            j+=1
        Y1_pred[k,:]=Y1_pred[k,:]/w
        m+=1
    print(Y1_pred)
    Y_pred=Y1_pred
else:
    Y_pred=Y_pred
        
        

# Compute confusion matrix
#cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
cnf_matrix =np.around(Y_pred,2)

np.set_printoptions(precision=1)
fig=plt.figure(figsize=(25,25))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix',cmap='Purples')

plt.show()
fig.savefig('Cross Sub cnf2')



#List=[]
#j=0
#for i in original:
#    c=y_pred[j]
#    l=C[c]
#    List.append(l);
#    j+=1
#print(List)  

RecognitionRate=0
for i in range(len(C)):
    RecognitionRate=RecognitionRate+Y_pred[i,i]
RecognitionRate=RecognitionRate/len(C)
print(RecognitionRate)
#
#from sklearn.metrics import classification_report,confusion_matrix
#import itertools 
#
#target_names = C
#					
#print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
#
#print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
#
#f=45
## Plotting the confusion matrix
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title,fontsize=40)
##    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45,fontsize=f)
#    plt.yticks(tick_marks, classes,fontsize=f)
#
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",fontsize=30,
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label',fontsize=f)
#    plt.xlabel('Predicted label',fontsize=f)
#
## Compute confusion matrix
##cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
#cnf_matrix =np.around(Y_pred,2)
#
#np.set_printoptions(precision=1)
#fig=plt.figure(figsize=(50,50))
#
## Plot non-normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names,
#                      title='Confusion Matrix',cmap='Greens')
#
#plt.show()
#fig.savefig('Cross Sub cnf')
#
#
#
#List=[]
#j=0
#for i in img_list:
#    c=y_pred[j]
#    l=C[c]
#    List.append(l);
#    j+=1
#print(List)  

#%%
##from skimage import io
#j=0
#for i in img_list:
#    label=List[j]
#    proba=np.amax(Y_pred[j,:])
#    label = "{}: {:.2f}%".format(label, proba * 100)
#    a=np.amax(Y_pred[j,:])
#    cv2.putText(original[j], label, (12, 24),  cv2.FONT_HERSHEY_TRIPLEX,1, (0,255, 255), 2)
##    cv2.putText(originalof[j], label, (12, 24),  cv2.FONT_HERSHEY_TRIPLEX,1, (0,255,255), 2)
#    cv2.imwrite(dataset+str(j)+'.jpg', original[j])
##    cv2.imwrite(dataset+"of"+str(j)+'.jpg', originalof[j])
#    cv2.waitKey(0)
#    j+=1
#%%
#L=List
#import pyttsx3
#engine  = pyttsx3.init()
#voices = engine.getProperty('voices')
#engine.setProperty('voice', voices[1].id) #change index to change voices
#engine.say(L)
#engine.runAndWait()