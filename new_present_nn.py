import random
import math

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from agent import Agent


n_t_set = 700
n_e_set = 300
epochs = 5

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', dest='mode', help='mode of the execution', choices=['t', 'e'],
						default='train', type=str)
	args = parser.parse_args()
	return args

def create_set(ammount, proportion):

	data = []
	labels = []

	label0 = 1
	label1 = 1

	for i in range(ammount):
		if i%2 == 0:
			x = random.randrange(1, 1000)
			y = math.sin(x/4) + math.sin(x+1) + math.cos(math.sin(x)) + 4 + math.log(x)
			label = [1, 0]

			inputs = [x, y]
			data.append(inputs)
			labels.append(label)

			label0 += 1
		else:
			x = random.randrange(1, 1000)
			y = math.sin(x)*math.cos((x^2)/10) + 2
			label = [0, 1]

			inputs = [x, y]
			data.append(inputs)
			labels.append(label)

			label1 += 1

	f = open("labels_creation.txt","w+")
	for label in labels:
		f.write(str(label))
	f.close()

	print(label0)
	print(label1)
	
	return data, labels

def main():
	mode = args.mode

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	agent = Agent()

	#1. Create tranning and evaluation sets
	t_set, t_labels = create_set(n_t_set, 50)
	e_set, e_labels = create_set(n_e_set, 50)


	#2. Trainning phase
	agent.train(t_set, t_labels, epochs)


	#3. Evaluate
	label0 = 0
	label1 = 0

	predicted0 = 0
	predicted1 = 0
	for i in range(100):
		if random.randrange(1, 100) > 70:
			x = random.randrange(1, 1000)
			y = math.sin(x/4) + math.sin(x+1) + math.cos(math.sin(x)) + 4 + math.log(x)
			label = 0
			label0 += 1
		else:
			x = random.randrange(1, 1000)
			y = math.sin(x)*math.cos((x^2)/10) + 2
			label = 1
			label1 += 1

		inputs = [x, y]

		output = agent.eval(inputs)

		#print(output[0].argmax(0).unsqueeze(0))
		if output == 0:
			predicted0 += 1
		else:
			predicted1 += 1

	print('=========================================')
	print('label0: ', label0)
	print('label1: ', label1)
	print('predicted0: ', predicted0)
	print('predicted1: ', predicted1)

if __name__ == "__main__":
	args = parse_args()
	main()