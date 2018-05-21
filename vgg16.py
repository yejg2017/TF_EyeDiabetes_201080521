import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import random_ops

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 64,
#                             """Number of images to process in a batch.""")

class Vgg16:
    def __init__(self,batch_size,num_example_per_epoch_for_train,model_path=None,learning_rate=1e-3, decay=0.95, decay_step=200, decay_factor=0.1,image_size=224,num_classes=5):
        # TODO load the model
        self.MOVING_AVERAGE_DECAY = decay              # The decay to use for the moving average.
        self.NUM_EPOCHS_PER_DECAY = decay_step         # Epochs after which learning rate decays.
        self.LEARNING_RATE_DECAY_FACTOR = decay_factor # Learning rate decay factor.
        self.INITIAL_LEARNING_RATE = learning_rate     # Initial learning rate.
        self.decay = decay
        self.image_size=image_size
        self.num_classes=num_classes
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=num_example_per_epoch_for_train
        self.batch_size=batch_size

    def inference(self, images):
        """Build the vgg16 model.

        Args:
            images: A tensor of shape [batch_size, image_size, image_size, channels]
            is_train: A tensor of shape [1] dtype tf.bool

        Returns:
            logits: Output tensor of the softmax
        """
        start_time = time.time()
        print('*************** build the model **************')
        assert images.get_shape().as_list()[1:] == [self.image_size, self.image_size, 3]
        # define the vgg16 model
        self.conv1_1 = self.conv_layer(images, [3,3,3,64], "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, [3,3,64,64], "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, [3,3,64,128], "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, [3,3,128,128], "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, [3,3,128,256], "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, [3,3,256,256], "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, [3,3,256,256], "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, [3,3,256,512], "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, [3,3,512,512], "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, [3,3,512,512], "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, [3,3,512,512], "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, [3,3,512,512], "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, [3,3,512,512], "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, 1000, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7,self.num_classes, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        print(("build model finished: %ds" % (time.time() - start_time)))

        return self.prob
        

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.

        Add summary for for "Loss" and "Loss/avg".
        
        Args:
            logits: Logits from inference().
            labels: Labels from 1-D tensor of shape [batch_size]

        Returns:
            Loss tensor of type float.
        """
        labels = tf.one_hot(labels,self.num_classes)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy, name='cross_entropy')
        # self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits), reduction_indices=[1]))
        return self.cross_entropy_mean

    def accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self, total_loss, global_step):
        """Train vgg model.

        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.

        Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps processed.
        
        Returns:
            train_op: optimizer for training.
        """
        num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)
         # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                         global_step,
                                         decay_steps,
                                         self.LEARNING_RATE_DECAY_FACTOR,
                                         staircase=True)
        tf.summary.scalar('learning_rate', lr)

        train_op = tf.train.RMSPropOptimizer(learning_rate=lr, decay=self.decay, epsilon=1e-10).minimize(self.cross_entropy)

        return train_op

    def max_pool(self, input_x, name):
        return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input_x, shape, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(shape=shape)

            conv = tf.nn.conv2d(input_x, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(shape=shape[-1])
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, input_x, hidden_dim,  name):
        with tf.variable_scope(name):
            shape = input_x.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input_x, [-1, dim])

            weights = self.get_fc_weight(shape=[dim, hidden_dim])
            biases = self.get_bias(shape=[hidden_dim])

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, shape):
        return tf.get_variable(name="filter", initializer=self.uniform_scaling(shape=shape))

    def get_bias(self, shape):
        return tf.Variable(tf.zeros(shape), name="biases")

    def get_fc_weight(self, shape):
        return tf.get_variable(name="weights", initializer=self.uniform_scaling(shape=shape))

    def uniform_scaling(self, shape=None, factor=1.0, dtype=tf.float32, seed=None):
        if shape:
            input_size = 1.0
            for dim in shape[:-1]:
                input_size *= float(dim)
            max_val = math.sqrt(3 / input_size) * factor
            return random_ops.random_uniform(shape, -max_val, max_val,
                                                dtype, seed=seed)
        else:
            return tf.uniform_unit_scaling_initializer(seed=seed, dtype=dtype)
