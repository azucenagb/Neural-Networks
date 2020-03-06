from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy:
	"""
	Class used for evaluation. Will take a single observation as input in the _call_ function and need to output the l6 dimensional logits for next action
	"""
	def __init__(self, model):
		self.model = model

	def __call__(self, obs):

		return self.model(obs[None, None, :, :, :]).view(-1)

class Model(nn.Module):
	def __init__(self, channels=[16, 32, 16], ks=5):
		super().__init__()

		self.l1 = nn.Linear(99, 5000)
		self.l2 = nn.Linear(5000, 6)
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(3, 32, 5, 2, 1)
		self.conv2 = nn.Conv2d(32, 64, 5, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
		self.fc = nn.Linear(6272, 6)
		self.fc2 = nn.Linear(3, 6)

		c0 = 6
		layers=[]
		self.width=1
		for c in channels:
			layers.append(nn.Conv1d(c0, c, ks))
			layers.append(nn.AvgPool1d(ks, 1, 2))
			layers.append(nn.LeakyReLU())
			c0 = c
			self.width += ks-1
		layers.append(nn.Conv1d(c0, 6, ks))
		self.width +=ks-1
		self.model=nn.Sequential(*layers)



	def forward(self, hist, hidden = None):
		'''
		Your code here
		Input size: (batch_size, sequence_length, channels, height, width)
		Output size: (batch_size, sequence_length, 6)
		'''
		bs = hist.size()[0]
		sl = hist.size()[1]
		c = hist.size()[2]
		h = hist.size()[3]
		w = hist.size()[4]

		hist = hist.view(-1, c, h, w)

		x = self.conv1(hist)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = x.view(-1, 6272)
		x = self.fc(x)
		x = x.view(bs, sl, -1)
		x = x.permute(0,2,1)
		x = F.pad(x, (self.width-1,0))
		output = self.model(x)
		output = output.permute(0,2,1)

		return output

	def policy(self):
		return Policy(self)
