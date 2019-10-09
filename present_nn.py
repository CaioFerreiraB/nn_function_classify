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

from model import MATH


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', dest='mode', help='mode of the execution', choices=['t', 'e'],
						default='train', type=str)
	args = parser.parse_args()
	return args

def train(neural_net, optimizer, criterion):
	epochs = 1000

	label0 = 0
	label1 = 0

	neural_net.train()

	losses = []
	correct = 0

	for epoch in range(epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		

		if random.randrange(1, 100) > 50:			
			x = random.randrange(1, 1000)
			y = math.sin(x/4) + math.sin(x+1) + math.cos(math.sin(x)) + 4 + math.log(x)
			#label = [1, 0]
			label = 0
			label0 += 1
		else:			
			x = random.randrange(1, 1000)
			y = math.sin(x)*math.cos((x^2)/10) + 2
			#label = [0, 1]
			label = 1
			label1 += 1

		inputs = [x, y]

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		inputs = torch.FloatTensor(inputs).unsqueeze(0)
		

		outputs = neural_net(inputs)
		#print(outputs)
		#print('===')
		output = outputs[0].argmax(0).unsqueeze(0).float()
		output.requires_grad = True
		#output = torch.sigmoid(outputs)
		#print(output)
		#print('--------------------')


		label = torch.FloatTensor([label]).unsqueeze(0)
		if output.item() == label.item():
			correct += 1
		"""
		#print(outputs)


		output = output.unsqueeze(0).float()
		output.requires_grad=True
		"""
		
		#print(output)
		print('label:', label)

		loss = criterion(output, label)
		print('loss: ', loss)

		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		losses.append(loss)

	plt.plot(losses)
	plt.ylabel('some numbers')
	plt.show()

	print('\ncorrect: ', correct, '\n')

	print('label0: ', label0)
	print('label1: ', label1)


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
		inputs = torch.FloatTensor(inputs).unsqueeze(0)

		output = neural_net(inputs)

		#print(output[0].argmax(0).unsqueeze(0))
		if output[0].argmax(0).unsqueeze(0).item() == 0:
			predicted0 += 1
		else:
			predicted1 += 1

	print('=========================================')
	print('label0: ', label0)
	print('label1: ', label1)
	print('predicted0: ', predicted0)
	print('predicted1: ', predicted1)
		



def main():
	mode = args.mode

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



	#1. Defining the  function

	# Define learning and trainning processes

	# Set neural network
	neural_net = MATH().to(device)

	# Set optimizer
	criterion = nn.BCELoss()
	optimizer = optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)

	# Trainning phase
	train(neural_net, optimizer, criterion)
	print('Finished Training')


	# evaluate model

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
		inputs = torch.FloatTensor(inputs).unsqueeze(0)

		output = neural_net(inputs)

		#print(output[0].argmax(0).unsqueeze(0))
		if output[0].argmax(0).unsqueeze(0).item() == 0:
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