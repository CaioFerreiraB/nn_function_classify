import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from model import Net


class Agent():
	def __init__(self):
		# Build network
		self.network = Net()

		# Create optimizer
		self.optimizer = optim.SGD(self.network.parameters(), lr=0.01)

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
		
		t_set = torch.FloatTensor(t_set).unsqueeze(0)
		t_labels = torch.FloatTensor([t_labels]).unsqueeze(0)

		for i in range(epochs):
			print(t_set)
			output = self.network.forward(t_set)

			loss = self.optimize(t_labels, output)