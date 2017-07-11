import unittest

import gym,ipdb
from nn_env import NnEnv
from time import sleep

class TestLeftRightEnv(unittest.TestCase):
	
    def test_env(self):
        env = NnEnv()
        env.reset()
        done=False
        while not done:
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            sleep(0.05)

        return True

def test_debug():
	env = NnEnv()
	env.seed(160)
	env.reset()
	done=False
	while not done:
	    #env.render()
	    action = env.action_space.sample()
	    #print action
	    observation, reward, done, info = env.step(action)
	    print "Env",observation, reward, done

if __name__ == '__main__':
	test_debug()
