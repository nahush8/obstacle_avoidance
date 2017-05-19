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

record = []
plt.ion()

class gp_prediction():
	def __init__(self):
		self.rbf_init_length_scale = np.array([1,1,1,1,1,1,1])
		self.kernel = C(1.0, (1e-3, 1e3)) * RBF(self.rbf_init_length_scale.shape, (1e-6, 1e6)) #C is a constant kernel and RBF is the squared exp kernel.
		
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
	epsilon = 0.2
	prev_length_of_record = 0
	game_obj = gameEngine.GameState()
	gp_obj = gp_prediction()
	plot_obj = plotting.plot_class()
	sum_of_reward_per_epoch = 0
	prev_state = [20,20,20,20,20,20]
	prev_state = np.array([prev_state])
	next_state = [[20,20,20,20,20,20]]
	timestr = time.strftime("%Y%m%d-%H%M%S")
	while True:
		if i != 0:
			randomNumber = random.random()
			if randomNumber >= epsilon:
				action = gp_obj.choose_action(next_state.tolist()[0])
			else:
				action = random.randint(0, 2)		
		elif i == 0:
			action = random.randint(0, 2)

		curr_reward, next_state = game_obj.frame_step(action)
		#time.sleep(0.2)
		newRecord = [prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]]
		if newRecord not in record:
			record.append(newRecord)
		#record.append([prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]])
		prev_state = next_state

		#print len(record)
		sum_of_reward_per_epoch += curr_reward
		if len(record) < 800:
			if abs(len(record) - prev_length_of_record) > 50:
				prev_length_of_record = len(record)
				plt.scatter(j,sum_of_reward_per_epoch)

				with open(timestr, 'a') as fp:
					fp.write(str(sum_of_reward_per_epoch) + '\n')
					fp.flush()
				fp.close()
				#plot_obj.plotting(record)
				gp_obj.gpq(record)
				print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
				sum_of_reward_per_epoch = 0
				j += 1
		else:
			if len(record) > 1200:
				epsilon = 0.05
			else:
				epsilon = 0.1

			if abs(len(record) - prev_length_of_record) > 100:
				prev_length_of_record = len(record)
				plt.scatter(j,sum_of_reward_per_epoch)

				with open(timestr, 'a') as fp:
					fp.write(str(sum_of_reward_per_epoch)+'\n')
					fp.flush()
				fp.close()
				#plot_obj.plotting(record)
				if len(record) < 1200:
					gp_obj.gpq(record)
				print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
				sum_of_reward_per_epoch = 0
				j += 1
		'''
		if curr_reward == -500:
			gp_obj.gpq(record)
			print len(record)
			#plot_obj.plotting(record)
			#plt.scatter(j,sum_of_reward_per_epoch)
			#for element in range (0,len(record)):
			print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
			sum_of_reward_per_epoch = 0
			
		'''
		i+= 1
		plt.pause(0.05)