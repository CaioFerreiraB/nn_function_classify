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
import numpy as np
import random
import copy

from agent import Agent

tam_population = 8
#---------------------------------------------------------------------------------
mutation_rate = 0.001 # perguntar
#---------------------------------------------------------------------------------
n_generations = 1000000
max_layers = 4 # max number of layers of the neural network
min_neurons = 2 # min number of neurons of a layer on the neural network
max_neurons = 30 # max number of neurons of a layer on the neural network
epochs = 10
n_trainings = 10 # MUST be higher or equal do 3 (n_trainings >= 3)
neurons_constant = 0.1


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

def init_population():
	individuals = []
	for i in range(tam_population):
		individual_in = np.append(np.array([2]), np.random.randint(low=min_neurons, high=max_neurons, size=random.randint(1,(max_layers-1))))
		individual_out = np.append(np.copy(individual_in[1:]),np.array([2])) 
	
		individuals.append([individual_in, individual_out])

	individuals = np.array(individuals)
		
	return individuals

def eval_population(individuals, fitness, t_set, t_labels, e_set, e_labels, device):
	print('EVALUATION -------------------------------------------------------------')

	for i in range(tam_population):
		individual  = individuals[i]

		fitness_vec = []
		for j in range(n_trainings):
			agent = Agent(len(individual[0]), individual[0], individual[1], device)

			agent.train(t_set, t_labels, epochs)

			false_positive, false_negative, true_positive, true_negative = agent.eval_batch(e_set, e_labels)

			fitness_vec.append(((true_positive + true_negative) - (false_positive + false_negative)) - ((sum(individual[0]) + 2)*neurons_constant)) 

			# another option to fitness function could be the hit rate of the network
			# fitness[i] = (true_positive + true_negative) / ((true_positive + true_negative) + (false_positive + false_negative))
			print('    -> training ', j, '| fitness: ', fitness_vec[j], '| individual: ', individual[0])

		fitness_sum = sum(fitness_vec) - min(fitness_vec) - max(fitness_vec)


		fitness[i] = fitness_sum/(n_trainings-2)
		print('-> individual ', i, 'fitness: ', fitness[i])


	print('-> fitness: ', fitness)

def elitism(fitness, individuals):
	print('ELITISM -------------------------------------------------------------')
	print(fitness)

	best_value = -1

	print('-> Getting best value')
	for j in range(tam_population):
		print('    -> Individual ', j)
		if (fitness[j] > best_value):
			best_value = fitness[j]
			best_index = j
	
	len_bi = len(individuals[best_index][0])
	
	print('-> Performing mutation and crossover')
	for i in range(tam_population):
		print('    -> Performing ', i)
		if(i != best_index):  #preservacao do melhor
			#crossover
			len_i = len(individuals[i][0])

			individual_aux = []
			individual_aux_in = []
			individual_aux_out = []

			if (len_i > len_bi):

				individual_aux_in = copy.copy(individuals[i,0][:len_bi]) + copy.copy(individuals[best_index,0])
				individual_aux_in =  np.append(individual_aux_in, copy.copy(individuals[i,0][len_bi:round((len_i - len_bi)/2)]))
				individual_aux_in = individual_aux_in//2


			if (len_i < len_bi):
				individual_aux_in = copy.copy(individuals[best_index,0][:len_i]) + copy.copy(individuals[i,0])
				individual_aux_in =  np.append(individual_aux_in, copy.copy(individuals[best_index,0][len_i:round((len_bi - len_i)/2)]))
				individual_aux_in = individual_aux_in//2

				

			if (len_bi == len_i):
				individual_aux_in = copy.copy(individuals[best_index,0]) + copy.copy(individuals[i,0])
				individual_aux_in = individual_aux_in//2

			#mutacao

			len_in = len(individual_aux_in)
			for i in range(1,len_in):
				min_add = min_neurons - individual_aux_in[i]
				max_add = max_neurons - individual_aux_in[i]
				individual_aux_in[i] += round(random.randint(min_add,max_add)*mutation_rate) 

			individual_aux_out = np.append(np.copy(individual_aux_in[1:]),np.array([2])) 

			individuals[i] = [copy.copy(individual_aux_in), copy.copy(individual_aux_out)]
		
	
	return best_value, best_index




def main():
	individuals = init_population()
	print(individuals)
	fitness = [None for _ in range(tam_population)]
	print('fitnes:\n', fitness)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('\n'+str(device)+'\n')


	best_value = 0


	n_e_set = args.n_e_set
	n_t_set = args.n_t_set
	

	#1. Create tranning and evaluation sets
	t_set, t_labels, t_label0, t_label1 = create_set(n_t_set)
	e_set, e_labels, e_label0, e_label1 = create_set(n_e_set)


	for i in range(n_generations):
		print('GENERATION: ', i)
		eval_population(individuals, fitness, t_set, t_labels, e_set, e_labels, device)
		best_value, best_index = elitism(fitness, individuals)
		f = open("best_value3.txt","a+")
		f.write(str(i) + ',' + str(best_value) + ',' + str(best_index) + ',' + str(individuals[best_index, 0]) + ',' + str(individuals[best_index, 1]) + '\n')
		f.close()
		print('RESULT -------------------------------------------------------------')
		print('best_value: ', best_value)
		print('individual: ', individuals[best_index])



	

if __name__ == "__main__":
	args = parse_args()
	main()