#!/usr/bin/python

import numpy as np
import math
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import pickle
import time
import gameEngine
import plotting

Q = {((2,2,2),0):0}
plt.ion()

class q_class():
	def __init__(self):
		self.alpha = 1
		self.gamma = 0.8
	#@profile
	def updateQ(self,state_x,action,reward,state_y):
		q_sa = Q.get((tuple(state_x),action), 0)
		max_q = 0
		for a in range(0,2):
			val = Q.get((tuple(state_y),a),0)
			if val > max_q:
				max_q = val
		newQ = q_sa + self.alpha * (reward + self.gamma * max_q - q_sa)
		Q[(tuple(state_x),action)] = newQ

	#@profile
	def choose_action(self,state_x):
		max_q = 0
		max_action = 0
		for a in range(0,2):
			val = Q.get((tuple(state_x),a),0)
			if val > max_q:
				max_q = val
				max_action = a

		return max_action		

if __name__ == "__main__":
	epsilon = 0.1	
	i = 0
	j = 0
	prev_length_of_record = 0
	game_obj = gameEngine.GameState()
	q_obj = q_class()
	plot_obj = plotting.plot_class()
	sum_of_reward_per_epoch = 0
	prev_state = [2,2,2]
	prev_state = np.array([prev_state])
	next_state = [[2,2,2]]
	timestr = time.strftime("%Y%m%d-%H%M%S")
	while True:
		if i != 0:
			randomNumber = random.random()
			if randomNumber >= epsilon:
				action = q_obj.choose_action(next_state.tolist()[0])
				#print action
			else:
				action = random.randint(0, 2)		
		elif i == 0:
			action = random.randint(0, 2)

		curr_reward, next_state = game_obj.frame_step(action)
		q_obj.updateQ(prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0])
		print Q
		prev_state = next_state
		'''
		sum_of_reward_per_epoch += curr_reward
		if abs(i - prev_length_of_record)> 100:
			prev_length_of_record = i
			plt.scatter(j,sum_of_reward_per_epoch)

			with open(timestr + '_q', 'a') as fp:
				fp.write(str(sum_of_reward_per_epoch) + '\n')
				fp.flush()
			fp.close()
			#plot_obj.plotting(record)
			print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
			sum_of_reward_per_epoch = 0
			j += 1
			#plot_obj.plotting(record)
		i+= 1
		plt.pause(0.05)
		'''