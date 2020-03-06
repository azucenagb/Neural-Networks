import torch.nn as nn

class MainLinear(nn.Module):
	def __init__(self):
		super(MainLinear, self).__init__()

		self.linear = nn.Linear(2, 2)

	def forward(self, x):
		x = self.linear(x)

		return x

class MainDeep(nn.Module):
	def __init__(self):
		super(MainDeep, self).__init__()
		self.fc1 = nn.Linear(2, 10)
		self.fc2 = nn.Linear(10, 5)
		self.fc3 = nn.Linear(5, 2)


	def forward(self, x):
		relu = nn.ReLU()
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		x = self.fc3(x)
		return x
