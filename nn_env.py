from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math, ipdb
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class NnEnv(gym.Env):
    def __init__(self, num_layers=2, num_hyperparams=2, num_filters=64, filter_height_vals=[1,3,5,7],
                filter_width_vals=[1,3,5,7], num_filters_vals=[24,36,48,64], strides=[1,2,3,4], epochs=1):

        # action and observation types
        self.action_space = spaces.Discrete(num_hyperparams)
        self.observation_space = spaces.MultiBinary(num_hyperparams)
        
        # general model architecture 
        self.num_layers=num_layers
        self.num_hyperparams=num_hyperparams
        
        # pre-set architecture parameters
        self.num_filters_vals=num_filters_vals
        self.strides=strides
        self.num_filters=num_filters

        # tunable architecture parameters
        self.filter_height_vals=filter_height_vals
        self.filter_width_vals=filter_width_vals

        
        # others
        self.epochs = epochs

        self._seed()
        self.load_data()
        self.reset()


    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        print (self.state)
        curr_layer,curr_hyper_parameter = self.state
            
        if curr_hyper_parameter == 0: # filter-height
            self.filter_height=self.filter_height_vals[action]
        elif curr_hyper_parameter == 1:  # filter-width
            self.filter_width=self.filter_width_vals[action]

        if self.state[1] == self.num_hyperparams: # layer finished
            # Add the layer to the model
            kernel_size=(self.filter_height, self.filter_width)
            if self.state[0]==0:                  # first-layer (have to specify input shape)
                self.model.add(Conv2D(self.num_filters, kernel_size=kernel_size, activation='relu', 
                    input_shape=self.input_shape))
            else:
                self.model.add(Conv2D(self.num_filters, kernel_size=kernel_size, activation='relu'))

            # Update the current layer, and set hyper-paramter to 0
            self.state[1]= 0 
            self.state[0]+= 1

            if self.state[0]==self.num_layers: # model architecture finished
                done = True
                self.add_fc_layers()
                self.train()
                reward=self.test()
            else:
                done = False
                reward=0
        else:
            self.state[1]+=1
            done = False
            reward=0

        observation=np.zeros(self.num_hyperparams)
        observation[action]=1
                
        return observation, reward, done, {}

    def _reset(self):
        self.state = [0,0] # [current layer, current hyper-parameter] 
        self.model = Sequential()
        return np.array(self.state)

    def add_fc_layers(self):
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def train(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))

    def test(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score[1]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self):
        pass
        #self.model_summary()

    def load_data(self):
        self.batch_size = 128
        self.num_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)



     