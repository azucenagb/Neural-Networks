import argparse, pickle, os
import torch
import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
import numpy as np

from .main import MainLinear, MainDeep


def train_linear(model):

#896 with 1956 got 0.0946
#1000 with 1956 got 0.0836
	inputs = torch.rand(972, 2)

	labels1 = ((torch.sqrt((inputs*inputs)[:,0] + (inputs*inputs)[:,1]))<=1).float()
	labels2 = ((torch.sqrt((inputs*inputs)[:,0] + (inputs*inputs)[:,1]))>1).float()
	labels = torch.rand(972, 2)
	labels[:,0] = labels1.view(-1)
	labels[:,1] = labels2.view(-1)

	model = MainLinear()
	criterion = nn.BCEWithLogitsLoss()

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	running_loss = 0.0

	epochs = 784
	for ep in range(epochs):
		for i in range(0,100):
			model.train()
			optimizer.zero_grad()
			outputs = (model(inputs))
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print('Epoch %d, loss:%.4f' % (ep+1, running_loss/100))
		running_loss = 0

	# Save the trained model
	dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
	save(model.state_dict(), os.path.join(dirname, 'linear')) # Do NOT modify this line


def train_deep(model):

	inputs = torch.rand(10000, 2)

	labels1 = ((torch.sqrt((inputs*inputs)[:,0] + (inputs*inputs)[:,1]))<=1).float()
	labels2 = ((torch.sqrt((inputs*inputs)[:,0] + (inputs*inputs)[:,1]))>1).float()
	labels = torch.rand(10000, 2)
	labels[:,0] = labels1.view(-1)
	labels[:,1] = labels2.view(-1)

	model = MainDeep()
	criterion = nn.BCEWithLogitsLoss()

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	running_loss = 0.0

	epochs = 250
	for ep in range(epochs):
		for i in range(0,100):
			model.train()
			optimizer.zero_grad()
			outputs = (model(inputs))
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print('Epoch %d, loss:%.4f' % (ep+1, running_loss/100))
		running_loss = 0

	# Save the trained model
	dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
	save(model.state_dict(), os.path.join(dirname, 'deep')) # Do NOT modify this line


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['linear', 'deep'])
	args = parser.parse_args()

	if args.model == 'linear':
		print ('[I] Start training linear model')
		train_linear(MainLinear())
	elif args.model == 'deep':
		print ('[I] Start training linear model')
		train_deep(MainDeep())

	print ('[I] Training finished')
