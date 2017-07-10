from __future__ import print_function
import math, ipdb
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from cnn import Model
from pprint import pprint

class NnEnv(gym.Env):

    def __init__(self, num_layers=2, num_hyperparams=2, num_hyperparams_vals=4,num_filters=64, filter_height_vals=[1,3,5,7],
                filter_width_vals=[1,3,5,7], num_filters_vals=[24,36,48,64], strides=[1,2,3,4], epochs=1,debug=False):

        # action and observation types
        self.action_space = spaces.Discrete(num_hyperparams_vals)
        self.observation_space = spaces.MultiBinary(num_hyperparams_vals)
        
        # general model architecture 
        self.num_layers=num_layers
        self.num_hyperparams=num_hyperparams
        self.num_hyperparams_vals=num_hyperparams_vals
        
        # untuned pre-set architecture parameters 
        self.num_filters_vals=num_filters_vals
        self.stride_height=1
        self.stride_width=1
        self.num_filters=num_filters

        # tunable architecture parameters
        self.filter_height_vals=filter_height_vals
        self.filter_width_vals=filter_width_vals

        # others
        self.epochs = epochs
        
        self.layers_params=[]
        self._seed()
        self.reset()
        self.debug=debug

    def _step(self, action):
        if self.debug:
            print ("action",action,"state-at-start",self.state)
        
        #check if action is in action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        curr_layer,curr_hyper_parameter = self.state
        
        if curr_hyper_parameter == 0: # filter-height
            self.filter_height=self.filter_height_vals[action]
            if self.debug:
                print ("filter_height",self.filter_height)

        elif curr_hyper_parameter == 1:  # filter-width
            self.filter_width=self.filter_width_vals[action]
            if self.debug:
                print ("filter_width",self.filter_width)

        if self.state[1] == self.num_hyperparams-1: # layer finished
            # Add the layer to the model
            self.layers_params.append({'filter_height':self.filter_height,'filter_width':self.filter_width,
                    'num_filters':self.num_filters,'stride_height':self.stride_height,
                    'stride_width':self.stride_width})
            # Update the current layer, and set hyper-paramter to 0
            self.state[1]= 0 
            self.state[0]+= 1

            if self.state[0]==self.num_layers: # model architecture finished
                done = True
                # build the model
                model=Model(epochs=self.epochs)
                model.build_model(self.layers_params)
                # train the model
                model.train()
                # test the model
                reward=model.test()
            else:                           # layers remaining
                done = False
                reward=0
        else:                               # model architecture remaining
            self.state[1]+=1
            done = False
            reward=0

        observation=np.zeros(self.num_hyperparams_vals)
        
        observation[action]=1
        if self.debug:
            print("State",self.state)

        return observation, reward, done, {}

    def _reset(self):
        self.state = [0,0] # [current layer, current hyper-parameter] 
        return np.array(self.state)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render(self, mode='human', close=False):
        pprint (self.layers_params)



     