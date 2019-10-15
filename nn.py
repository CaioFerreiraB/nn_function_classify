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


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', dest='mode', help='mode of the execution', choices=['t', 'e'],
						default='train', type=str)
	parser.add_argument('-nts', '--n_t_set', dest='n_t_set', help='number of elements in trainning set',
						default=700, type=int)
	parser.add_argument('-nes', '--n_e_set', dest='n_e_set', help='number of elements in evaluation set',
						default=300, type=int)
	parser.add_argument('-nl', '--n_layers', dest='n_layers', help='number of layers on the neural network',
						default=4, type=int)
	parser.add_argument('-e', '--epochs', dest='epochs', help='number of epochs for trainning phase',
						default=5, type=int)
	parser.add_argument('-in','--in_neurons', dest='in_neurons', nargs='+', help='list of in neurons',
						default=[2, 100, 300, 100], type=int)
	parser.add_argument('-on','--out_neurons', dest='out_neurons', nargs='+', help='list of out neurons',
						default=[100, 300, 100, 2], type=int)


	args = parser.parse_args()
	return args

def create_set(ammount):
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
	
	return data, labels, label0, label1



def main():
	mode = args.mode
	in_neurons = args.in_neurons
	out_neurons = args.out_neurons
	epochs = args.epochs
	n_layers = args.n_layers
	n_e_set = args.n_e_set
	n_t_set = args.n_t_set

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	agent = Agent(n_layers, in_neurons, out_neurons)

	#1. Create tranning and evaluation sets
	t_set, t_labels, t_label0, t_label1 = create_set(n_t_set)
	e_set, e_labels, e_label0, e_label1 = create_set(n_e_set)


	#2. Trainning phase
	agent.train(t_set, t_labels, epochs)


	#3. Evaluate
	false_positive, false_negative, true_positive, true_negative = agent.eval_batch(e_set, e_labels)

	#4.

	print('e_label0: ', e_label0)
	print('e_label1: ', e_label1)
	print('false_positive: ', false_positive)
	print('false_negative: ', false_negative)
	print('true_positive: ', true_positive)
	print('true_negative: ', true_negative)
	print('hit_rate: ', (true_negative+true_positive)/((true_negative+true_positive)+(false_negative+false_positive)))

	

if __name__ == "__main__":
	args = parse_args()
	main()