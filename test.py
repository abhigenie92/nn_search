import unittest

import gym
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


if __name__ == '__main__':
	env = NnEnv()
	env.reset()
	done=False
	while not done:
	    #env.render()
	    action = env.action_space.sample()
	    print action
	    observation, reward, done, info = env.step(action)
	    print observation, reward, done
	    sleep(0.05)
	unittest.main()