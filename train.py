from __future__ import division
#from matplotlib.pyplot import  imshow
import data_utils as du
import os
import pickle
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
import tensorflow as tf
import pandas as pd
from time import  time
from vgg16 import Vgg16
import warnings

# ignore warning
warnings.filterwarnings("ignore")


datadir='/home/ye/user/yejg/database/Kaggle_Eye/train_001/train'
label_file='/home/ye/user/yejg/database/Kaggle_Eye/train_001/trainLabels.csv'

files,labels=du.get_files(basedir=datadir,label_file=label_file,shuffle=True)




# Data iteration
sss = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.2, random_state=123)
train_idx, valid_idx = next(iter(sss))

train_files = np.array(files)[train_idx].tolist()  # some sort of hack..
train_labels = [labels[i] for i in train_idx]
n_train = len(train_files)
print('Training set size: {}'.format(n_train))
coefs = [0, 7, 3, 22, 25]

train_files, train_labels = du.oversample_set(train_files, train_labels, coefs)
#train_labels = du.one_hot(train_labels)
n_train = len(train_files)
assert len(train_files) == train_labels.shape[0]
print('Oversampled set size: {}'.format(len(train_files)))


valid_files = np.array(files)[valid_idx].tolist()  # some sort of hack..
valid_labels = [labels[i] for i in valid_idx]
n_valid = len(valid_files)
print('Validation set size: {}'.format(n_valid))


#  parameters
base_learning=1e-3
decay_rate=0.95
decay_step=200
num_example_per_epoch_for_train=len(train_files)
image_size=224
num_classes=5
max_epoch=100
print_step=10
save_step=50
val_step=50
batch_size=32


features=open('./features/images_features.pkl','rb')
mean_std=pickle.load(features)
mean,std=mean_std['mean'],mean_std['std']
print('mean of images:',mean,'\n')
print('std of images:',std,'\n')


# BatchIterator
train_iter = du.BatchIterator(train_files,
                           train_labels,
                           batch_size,
                           normalize=(mean, std),
                           process_func=du.parallel_augment)
valid_iter = du.BatchIterator(valid_files,
                           valid_labels,
                           batch_size,
                           normalize=(mean, std),
                           process_func=du.parallel_augment,
                           testing=True)
# Transform batchiterator to a threaded iterator
train_iter = du.threaded_iterator(train_iter)
valid_iter = du.threaded_iterator(valid_iter)


X=tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,3])
y=tf.placeholder(tf.int64,shape=[batch_size])
global_step = tf.Variable(0, trainable=False)


vgg16=Vgg16(batch_size=batch_size,image_size=image_size,learning_rate=base_learning,decay=decay_rate,decay_step=decay_step,
           num_example_per_epoch_for_train=num_example_per_epoch_for_train,num_classes=num_classes)
vgg16.inference(X)

prob=vgg16.prob
logits=vgg16.fc8
loss=vgg16.loss(logits=logits,labels=y)
accuracy=vgg16.accuracy(logits=logits,labels=y)
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



n_train_batch=len(train_files)//batch_size
n_val_batch=len(valid_files)//batch_size
print(n_train_batch,n_val_batch)


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
        for iter in range(n_train_batch):
            tra_x,tra_y=next(train_iter)
            tra_x=tra_x.transpose(0,2,3,1)
            
            # start iteration
            iter_start = time()
            sess.run([train_op],feed_dict={X:tra_x,y:tra_y})
            t_acc,t_loss,merged=sess.run([accuracy,loss,summary_op],feed_dict={X:tra_x,y:tra_y})
            tra_acc.append(t_acc)
            tra_loss.append(t_loss)
            
            if (iter+1)%print_step==0:
                print('[epoch:%d/iter:%d]  minibatch(size:%d) train_loss: %.2f,accuracy: %.2f %% - Elapsed time: %.2fs' % (epoch, iter + 1,batch_size, t_loss,t_acc * 100., time() - iter_start))
                train_writer.add_summary(merged,(iter+1))
            
            if (iter+1)%val_step==0:
                val_acc=[]
                val_loss=[]
                for val_s in range(n_val_batch):
                    val_x,val_y=next(valid_iter)
                    val_x=val_x.transpose(0,2,3,1)
                    v_acc,v_loss,val_merged=sess.run([accuracy,loss,summary_op],feed_dict={X:val_x,y:val_y})
                    val_acc.append(v_acc)
                    val_loss.append(v_loss)
                    val_writer.add_summary(val_merged,(iter+1))
                    
                val_acc_avg=np.mean(val_acc)
                val_loss_avg=np.mean(val_loss)
                
                print('Validation --> [epoch:%d minibatch %i/%i, loss: %.2f,accuracy: %.2f %%' % (epoch,iter+1, n_val_batch, val_loss_avg,val_acc_avg * 100.))
                
        tra_acc_avg=np.mean(tra_acc)
        tra_loss_avg=np.mean(tra_loss)
        print('Train-->[epoch:%d]-----mean loss:%.2f,mean accuracy:%.2f%%'%(epoch,tra_loss_avg,tra_acc_avg))
        
        if (epoch+1)%save_step==0 or (epoch+1)==max_epoch:
            saver.save(sess, checkpoint_dir, global_step=step)

