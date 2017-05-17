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

record = []

class gp_prediction():

	def findMax(self,gp,next_state):
		tempMu = 0
		arrayList = []
		for x in (0,2):
			test = next_state + [x]
			arrayList.append(test)
		arrayList = np.array(arrayList)
		tempMu,sigma = gp.predict(arrayList, return_std=True, return_cov=False) 

		return max(tempMu)

	def gpq(self,record):
		inputX = []
		outputY = []

		kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-1, 1e1)) #C is a constant kernel and RBF is the squared exp kernel.
		
		gp = GaussianProcessRegressor(kernel=kernel,optimizer='fmin_l_bfgs_b' ,n_restarts_optimizer=9,alpha=1e-2)
		for elements in record:
			#for element in range (0,len(record)):
			inputX.append(elements[0] + [elements[1]])
			outputY.append((elements[2] +self.findMax(gp,elements[3])))
			#print inputX
		#print outputY
		dX = np.array(inputX)
		#print outputY
		tX = np.array(outputY)
		#st = time.time()
		gp.fit(dX,tX)


if __name__ == "__main__":
	game_obj = gameEngine.GameState()
	gp_obj = gp_prediction()
	prev_state = [[20,20,20,20,20,20]]
	while True:
		action = random.randint(0, 2)
		curr_reward, next_state = game_obj.frame_step(action)
		time.sleep(0.1)
		prev_state = next_state
		record.append([prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]])
		print len(record)
		if curr_reward == -500:
			gp_obj.gpq(record)
