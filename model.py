
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import time

class Net(nn.Module):

	def __init__(self, n_layers, in_neurons, out_neurons):
		super(Net, self).__init__()

		self.layers = nn.ModuleList()

		for i in range(n_layers):
			self.layers.append(nn.Linear(in_neurons[i], out_neurons[i]))
			
	def forward(self, x, n_layers):
		for i in range(n_layers):
			x = self.layers[i](x)

		return x

	def save(self, path, step, optimizer):
		torch.save({
			'step': step,
			'state_dict': self.state_dict(),
			'optimizer': optimizer.state_dict()
		}, path)
			
	def load(self, checkpoint_path, optimizer=None):
		print('LOAD PATH	--  model.load:', checkpoint_path)

		checkpoint = torch.load(checkpoint_path)
		step = checkpoint['step']
		self.load_state_dict(checkpoint['state_dict'])
		if optimizer is not None:
			optimizer.load_state_dict(checkpoint['optimizer'])