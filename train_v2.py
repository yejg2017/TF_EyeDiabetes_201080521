from __future__ import division
import pickle
import pandas as pd
from vgg16 import Vgg16
import numpy as np
import os
#import data_utils as du
#from data_utils import *
from sklearn.cross_validation import StratifiedShuffleSplit
import tensorflow as tf
import warnings
from sklearn import metrics
from Dataloader import *
# ignore warning
warnings.filterwarnings("ignore")


# load  data
train_dir='/home/ye/user/yejg/LEARN/DL_MODEL_LEARN/TensorFlow/TransferLearn/Fine_tune/data/train.txt'
val_dir='/home/ye/user/yejg/LEARN/DL_MODEL_LEARN/TensorFlow/TransferLearn/Fine_tune/data/val.txt'

##
print('load image list...')
tra_img_list=pd.read_csv(train_dir,header=None,sep=' ')
tra_files=[str(x) for x in tra_img_list.iloc[:,0]]
tra_labels=[int(x) for x in tra_img_list.iloc[:,1]]

val_img_list=pd.read_csv(val_dir,header=None,sep=' ')
val_files=[str(x) for x in val_img_list.iloc[:,0]]
val_labels=[int(x) for x in val_img_list.iloc[:,1]]
####

tra_images_list=np.array([tra_files,tra_labels])
tra_images_list=np.transpose(tra_images_list)
np.random.shuffle(tra_images_list)
print(tra_images_list.shape)

tra_files=[str(x) for x in tra_images_list[:,0]]
tra_labels=[int(x) for x in tra_images_list[:,1]]

###
val_images_list=np.array([val_files,val_labels])
val_images_list=np.transpose(val_images_list)
np.random.shuffle(val_images_list)
print(val_images_list.shape)

val_files=[str(x) for x in val_images_list[:,0]]
val_labels=[int(x) for x in val_images_list[:,1]]

###

print('load hsv,rgb features...')
f=open('/home/ye/user/yejg/database/Kaggle_Eye/features/features.pkl','rb')
features=pickle.load(f)
global_mean_rgb,global_mean_hsv=features['global_mean_rgb'],features['global_mean_hsv']
print(global_mean_hsv)
print(global_mean_rgb)


#  parameters
base_learning=1e-3
decay_rate=0.95
decay_step=200
num_example_per_epoch_for_train=len(tra_images_list)
image_size=224
num_classes=5
max_epoch=100
print_step=10
save_step=50
val_step=50
batch_size=20
method='enforce'
#
print('create data iterator...')
train_iter=BatchIter(image_list=tra_images_list,batch_size=batch_size,ishandle=True,
                     features=(global_mean_hsv,global_mean_rgb),shuffle=True,method=method)

val_iter=BatchIter(image_list=tra_images_list,batch_size=batch_size,ishandle=False,
                     features=(global_mean_hsv,global_mean_rgb),shuffle=False,method=method)


# 
X=tf.placeholder(tf.float32,shape=[None,image_size,image_size,3])
y=tf.placeholder(tf.int64,shape=[None])
global_step = tf.Variable(0, trainable=False)

#
print('build model...')
vgg16=Vgg16(batch_size=batch_size,image_size=image_size,learning_rate=base_learning,decay=decay_rate,decay_step=decay_step,
           num_example_per_epoch_for_train=num_example_per_epoch_for_train,num_classes=num_classes)
vgg16.inference(X)

prob=vgg16.prob
logits=vgg16.fc8
loss=vgg16.loss(logits=logits,labels=y)
accuracy=vgg16.accuracy(logits=logits,labels=y)
pred=vgg16.prediction()
train_op=vgg16.train(total_loss=loss,global_step=global_step)


# path
checkpoint_dir='./checkpoint'
tensorboard_dir='./tensorboard'
logs_train='./tensorboard/train'
logs_val='./tensorboard/val'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)
    
if not os.path.exists(logs_train):
    os.mkdir(logs_train)
    
if not os.path.exists(logs_val):
    os.mkdir(logs_val)


n_batch=len(train_iter)//batch_size
n_val_batch=len(val_iter)//batch_size
print('number of train batch:%d'%n_batch,'\n',
      'number of val batch:%d'%n_val_batch)
print('Traing........\n')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver(tf.global_variables())
    summary_op=tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val, sess.graph)
    
    for epoch in range(max_epoch):
        tra_acc=[]
        tra_loss=[]
        for iter in range(n_batch):
            tra_x,tra_y=train_iter.next()
            iter_start = time()
            
            sess.run([train_op],feed_dict={X:tra_x,y:tra_y})
            prediction,t_acc,t_loss,merged=sess.run([pred,accuracy,loss,summary_op],feed_dict={X:tra_x,y:tra_y})
            tra_acc.append(t_acc)
            tra_loss.append(t_loss)
            
            if (iter+1)%print_step==0:
                print('[epoch:%d/iter:%d]  minibatch(size:%d) train_loss: %.2f,accuracy: %.2f %%--Elapsed time: %.2fs' % (epoch, iter + 1,n_batch, t_loss,t_acc * 100.,
                                 time() - iter_start))
                print('---precision:%.4f,f1_score:%.4f,recall_score:%.4f---'%(metrics.precision_score(tra_y,prediction,average='macro'
                           ),metrics.f1_score(tra_y,prediction,average='macro'),metrics.recall_score(tra_y,prediction,average='macro')))

            if (iter+1)%val_step==0:
                val_acc=[]
                val_loss=[]
                for val_s in range(n_val_batch):
                    val_x,val_y=val_iter.next()
                    v_acc,v_loss,val_merged=sess.run([accuracy,loss,summary_op],feed_dict={X:val_x,y:val_y})
                    val_acc.append(v_acc)
                    val_loss.append(v_loss)
                    val_writer.add_summary(val_merged,(iter+1))

                val_acc_avg=np.mean(val_acc)
                val_loss_avg=np.mean(val_loss)

                print('Validation --> [epoch:%d/iter:%d] minibatch (size:%d), val_loss: %.2f,accuracy: %.2f %%' % (epoch,iter+1, n_val_batch, val_loss_avg,val_acc_avg * 100.))
                

        tra_acc_avg=np.mean(tra_acc)
        tra_loss_avg=np.mean(tra_loss)
        print('Train-->[epoch:%d]-----mean loss:%.2f,mean accuracy:%.2f%%'%(epoch,tra_loss_avg,tra_acc_avg))

        if (epoch+1)%save_step==0 or (epoch+1)==max_epoch:
            saver.save(sess, checkpoint_dir, global_step=step)

