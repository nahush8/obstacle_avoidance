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

LASER_MAX_VAL = 10
numOfActions = 4
numOfLasers = 3
record = []
cache = {(LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL):2}
plt.ion()

class gp_prediction():
	def __init__(self):
		#self.rbf_init_length_scale = np.array([1,1,1,1,1,1,1])
		self.rbf_init_length_scale = np.array([1,1,1,1])
		self.kernel = C(1.0, (1e-3, 1e3)) * RBF(self.rbf_init_length_scale.shape, (1e-5, 1e5)) #C is a constant kernel and RBF is the squared exp kernel.
		
		self.gp = GaussianProcessRegressor(kernel=self.kernel,optimizer='fmin_l_bfgs_b' ,n_restarts_optimizer=9,alpha=1e-2)
		self.gamma = 0.8

	def set_gp(self,gp):
		self.gp = gp
	#@profile
	def findMax(self,next_state):
		arrayList = []
		for x in range(0,numOfActions):
			test = next_state + [x]
			arrayList.append(test)
		arrayList = np.array(arrayList)
		tempMu,sigma = self.gp.predict(arrayList, return_std=True, return_cov=False) 
		return max(tempMu)

	#@profile
	def gpq(self,record):
		inputX = []
		outputY = []
		
		for elements in record:
			#for element in range (0,len(record)):
			inputX.append(elements[0] + [elements[1]])
			outputY.append((elements[2] + self.gamma * self.findMax(elements[3])))
			#print inputX

		dX = np.array(inputX)
		#print outputY
		tX = np.array(outputY)
		#st = time.time()
		print "DOING GP FIT"
		self.gp.fit(dX,tX)
		with open('gp_june17_env3', 'wb') as fp:
			pickle.dump(self.gp, fp)
		fp.close()
	#@profile
	def choose_action(self,next_state):
		tempMu = 0
		arrayList = []
		listMu = []
		action_value = 2
		#if len(record) > 1200:
		ret = cache.get(tuple(next_state),-999)
		#else:
		if ret != -999:
			return ret
		else:
			for x in range(0,numOfActions):
				test = next_state + [x]
				arrayList.append(test)
			arrayList = np.array(arrayList)
			tempMu,sigma = self.gp.predict(arrayList, return_std=True, return_cov=False) 
			listMu = list(tempMu)
			maxIndex  = listMu.index(max(listMu))
			tempList = arrayList[maxIndex]
			#print tempList
			action_value = tempList[numOfLasers]
			#print action_value
			#if len(record) > 1200:
			cache[tuple(next_state)] = action_value
			return action_value

if __name__ == "__main__":

	i = 0
	j = 0
	itr = 0	
	epsilon = 0.1
	good_epoch = 0
	epoch = 0
	iteration = 0
	average_sum_of_rewards = 0
	prev_epoch_reward = 0
	net_sum_of_rewards = 0
	prev_length_of_record = 0
	game_obj = gameEngine.GameState()
	gp_obj = gp_prediction()
	sum_of_reward_per_epoch = 0
	prev_state = [LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL]
	prev_state = np.array([prev_state])
	next_state = [[LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL]]
	timestr = time.strftime("%Y%m%d-%H%M%S")
	maxEpoch = 6
	record_for_epoch = []
	
	while epoch < 1000:
		action = random.randint(0, numOfActions-1)
		curr_reward, next_state = game_obj.frame_step(action)