import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
import numpy as np

import time
import matplotlib.pyplot as plt

from model import Net


class Agent():
	def __init__(self, n_layers, in_neurons, out_neurons):
		# Build network
		self.network = Net(n_layers, in_neurons, out_neurons)
		self.n_layers = n_layers

		# Create optimizer
		self.optimizer = optim.SGD(self.network.parameters(), lr=0.0001)

	def optimize(self, target, values):
		self.optimizer.zero_grad()
		
		# Compute Huber loss
		loss = F.smooth_l1_loss(values, target)
		loss_return = loss.item()

		# Optimize the model
		loss.backward()
		self.optimizer.step()

		return loss_return

	def train(self, t_set, t_labels, epochs):
		self.network.train()

		losses = []

		for i in range(epochs):

			t_set, t_labels = shuffle(t_set, t_labels)

			j = 0
			for t in t_set:
				t = torch.FloatTensor(t)

				output = self.network.forward(t, self.n_layers)				

				label = torch.FloatTensor(t_labels[j])
				j += 1

				loss = self.optimize(label, output)
				losses.append(loss)

		return losses

	def eval(self, inputs):
		self.network.train()
		inputs = torch.FloatTensor(inputs)
		
		output = self.network.forward(inputs, self.n_layers)

		return output.argmax(0).unsqueeze(0).item()

	def eval_batch(self, e_set, e_label):
		self.network.train()

		false_positive = 0
		false_negative = 0
		true_positive = 0
		true_negative = 0

		right = 0
		wrong = 0
		i = 0

		for inputs in e_set:
			inputs = torch.FloatTensor(inputs)
			output = self.network.forward(inputs, self.n_layers)
			output = output.argmax(0).unsqueeze(0).item()
			
			label = np.asarray(e_label[i]).argmax()
			i += 1

			if output == 0 and label == 0: 
				true_positive +=1
				right += 1
			elif output == 0 and label == 1:
				false_positive +=1
				wrong += 1
			elif output == 1 and label == 1: 
				true_negative +=1
				right += 1
			elif output == 1 and label == 0: 
				false_negative +=1
				wrong += 1

		return false_positive, false_negative, true_positive, true_negative