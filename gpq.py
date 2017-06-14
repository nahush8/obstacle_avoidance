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
import gameEngine2
import plotting

LASER_MAX_VAL = 10
numOfActions = 4
numOfLasers = 3
record = []
cache = {(LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL):2}
epoch = 0
iteration = 0
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
		with open('gp_june14_env4', 'wb') as fp:
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
	prev_length_of_record = 0
	game_obj = gameEngine2.GameState()
	gp_obj = gp_prediction()
	sum_of_reward_per_epoch = 0
	prev_state = [LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL]
	prev_state = np.array([prev_state])
	next_state = [[LASER_MAX_VAL,LASER_MAX_VAL,LASER_MAX_VAL]]
	timestr = time.strftime("%Y%m%d-%H%M%S")
	
	while epoch < 1000:
		if i != 0:
			randomNumber = random.random()
			if randomNumber >= epsilon:
				action = gp_obj.choose_action(next_state.tolist()[0])
			else:
				action = random.randint(0, numOfActions-1)		
		elif i == 0:
			action = random.randint(0, numOfActions-1)

		curr_reward, next_state = game_obj.frame_step(action)
		iteration = iteration + 1
		newRecord = [prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]]
		if newRecord not in record:
			record.append(newRecord)
		#record.append([prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]])
		prev_state = next_state
		'''
		if abs(len(record)-prev_length_of_record) > 200:
			prev_length_of_record = len(record)	
			if len(record) < 1200:
				gp_obj.gpq(record)
			else:
				print "NO MORE FITTING !!"
		'''
		sum_of_reward_per_epoch += curr_reward
		#print len(record)
		if iteration % 200 == 0:
			
			#prev_length_of_record = len(record)		
			#plt.scatter(epoch,sum_of_reward_per_epoch)
			
			with open(timestr + '_gpq', 'a') as fp:
				fp.write(str(sum_of_reward_per_epoch) + '\n')
				fp.flush()
			fp.close()
			
			#plot_obj.plotting(record)
			#print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
			sum_of_reward_per_epoch = 0
			epoch += 1
			if epoch < 6:
				gp_obj.gpq(record)
			else:
				print "NO MORE FITTING !!"
			#plot_obj.plotting(record)
		i += 1
		#plt.pause(0.05)
		
	
	'''
	with open ('gp_june9_test', 'rb') as fp:
			gp = pickle.load(fp)
	
	gp_obj.set_gp(gp)
	while True:
		if i != 0:
			randomNumber = random.random()
			if randomNumber >= epsilon:
				action = gp_obj.choose_action(next_state.tolist()[0])
			else:
				action = random.randint(0, numOfActions-1)		
			#action = gp_obj.choose_action(next_state.tolist()[0])
		else:
			action = random.randint(0, 3)
		curr_reward, next_state = game_obj.frame_step(action)
		epoch = epoch + 1
		#newRecord = [prev_state.tolist()[0],action,curr_reward,next_state.tolist()[0]]
		#if newRecord not in record:
		#record.append(newRecord)
		prev_state = next_state
		

		sum_of_reward_per_epoch += curr_reward

		if epoch % 200 == 0:
			#prev_length_of_record = len(record)
			#plt.scatter(j,sum_of_reward_per_epoch)

			with open(timestr + '_gpq', 'a') as fp:
				fp.write(str(sum_of_reward_per_epoch) + '\n')
				fp.flush()
			#fp.close()
			#plot_obj.plotting(record)
			print 'REWARD COLLECTED THIS EPOCH: %d' % sum_of_reward_per_epoch
			sum_of_reward_per_epoch = 0
			#j += 1
			#plot_obj.plotting(record)
		
		i += 1
		#plt.pause(0.05)	
	'''
	'''
	arrayList = []
	listMu = []
	heat = np.zeros((1000, 4))
	
	print len(states)
	with open ('gp', 'rb') as fp:
			gp = pickle.load(fp)
	print "GP loaded"
	for s in states:
		for a in range(0,4):
			test = s + [a]
			arrayList.append(test)
	arrayList = np.array(arrayList)
	tempMu, sigma = gp.predict(arrayList, return_std=True, return_cov=False) 
	listMu = list(tempMu)
	'''

	'''
	fig, ax = plt.subplots(figsize=(10,10))  
	heat = np.zeros((64, 4))

	#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	heat = np.zeros((1000, 4))
	with open('listMu', 'rb') as fp:
		listMu = pickle.load(fp)
	fp.close()
	states = []
	for i in range(1,11):
		for j in range(1,11):
			for k in range(1,11):
				states.append([i,j,k])
	for s in states:
		for a in range(0,numOfActions):
			heat[itr][a] = listMu[itr]
		itr+=1
	itr = 0	
	plt.imshow(heat, cmap='coolwarm', interpolation='nearest',aspect='auto')
	
	'''
	'''
	for s in states:
		heat[itr][1] = listMu[itr]
		itr+=1
	itr = 0	
	ax2.imshow(heat, cmap='coolwarm', interpolation='nearest',aspect='auto')
	for s in states:
		heat[itr][2] = listMu[itr]
		itr+=1
	itr = 0	
	ax3.imshow(heat, cmap='coolwarm', interpolation='nearest',aspect='auto')
	for s in states:
		heat[itr][3] = listMu[itr]
		itr+=1
	itr = 0	

	ax4.imshow(heat, cmap='coolwarm', interpolation='nearest',aspect='auto')
	
	
	plt.colorbar()
	plt.show()
	'''