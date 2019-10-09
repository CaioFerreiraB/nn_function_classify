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
	def __init__(self):
		# Build network
		self.network = Net()

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

		f = open("labels_train.txt","w+")

		for i in range(epochs):

			t_set, t_labels = shuffle(t_set, t_labels)

			j = 0
			class0 = 0
			class1 = 0
			right0 = 0
			right1 = 0
			for t in t_set:
				t = torch.FloatTensor(t)

				output = self.network.forward(t)				

				label = torch.FloatTensor(t_labels[j])
				f.write(str(label))

				loss = self.optimize(label, output)
				losses.append(loss)


				if label[0].item() == 1:
					class0 += 1
					if output.argmax(0).unsqueeze(0).item() == 0:
						right0 += 1
				if label[0].item() == 0:
					class1 += 1
					if output.argmax(0).unsqueeze(0).item() == 1:
						right1 += 1

				j += 1


		print('classe0:', class0)
		print('classe1:', class1)
		print('right0:', right0)
		print('right1:', right1)

		return losses

	def eval(self, inputs):
		self.network.train()
		inputs = torch.FloatTensor(inputs)
		
		output = self.network.forward(inputs)

		return output.argmax(0).unsqueeze(0).item()