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
import matplotlib.pyplot as plt

record = []
plt.ion()

class gp_prediction():
	def __init__(self):

		self.kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-1, 1e1)) #C is a constant kernel and RBF is the squared exp kernel.
		
		self.gp = GaussianProcessRegressor(kernel=self.kernel,optimizer='fmin_l_bfgs_b' ,n_restarts_optimizer=9,alpha=1e-2)

	#@profile
	def findMax(self,next_state):
		tempMu = 0
		arrayList = []
		for x in (0,2):
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
			outputY.append((elements[2] +self.findMax(elements[3])))
			#print inputX
		#print outputY
		dX = np.array(inputX)
		#print outputY
		tX = np.array(outputY)
		#st = time.time()
		print "DOING GP FIT"
		self.gp.fit(dX,tX)

	#@profile
	def choose_action(self,next_state):
		tempMu = 0
		arrayList = []
		listMu = []
		action_value = 0
		for x in (0,2):
			test = next_state + [x]
			arrayList.append(test)
		arrayList = np.array(arrayList)
		tempMu,sigma = self.gp.predict(arrayList, return_std=True, return_cov=False) 
		listMu = list(tempMu)
		maxIndex  = listMu.index(max(listMu))
		tempList = arrayList[maxIndex]
		#print tempList
		action_value = tempList[6]
		#print action_value 
		return action_value

if __name__ == "__main__":
	i = 0
	j = 0
	epsilon = 0.1
	game_obj = gameEngine.GameState()
	gp_obj = gp_prediction()
	sum_of_reward_per_epoch = 0
	prev_state = [20,20,20,20,20,20]
	prev_state = np.array([prev_state])
	next_state = [[20,20,20,20,20,20]]
	while True:
		if i != 0:
			randomNumber = random.random()
			if randomNumber >= epsilon:
				action = gp_obj.choose_action(next_state.tolist()[0])
				print action
			else:
				action = random.randint(0, 2)
		
		elif i == 0:
			action = random.randint(0, 2)

		curr_reward, next_state = game_obj.frame_step(action)
		#time.sleep(0.1)
		record.append([prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]])
		prev_state = next_state
		#print len(record)
		sum_of_reward_per_epoch += curr_reward
		if curr_reward == -500:
			gp_obj.gpq(record)
			plt.scatter(j,sum_of_reward_per_epoch)
			print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
			sum_of_reward_per_epoch = 0
			j += 1
		i+= 1
		plt.pause(0.05)