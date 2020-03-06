import torch

class Main(torch.nn.Module):
	def forward(self, x):
		# The input x is a series of random numbers of size k x 2

		# The dist has size k x 1 and holds the distances of each random point of the input to the point [0,0],
		# i.e. the distance of x[i,:] to [0,0] is held in dist[i] where 0<=i<=k
		dist = torch.sqrt((x*x)[:,0] + (x*x)[:,1])

		# The ind has size k x 1 and indicates whether or not each random point is inside the unit circle,
		# i.e. ind[i] is 1 if the point x[i,:] is inside the unit circle.
		ind = dist<1

		# The function returns an approximation of pi by calculating the ratio of points inside the unit circle.
		# We multiply the ratio obtained by four since we were only using one quadrant of the circle.
		pi = torch.mean(ind.float())*4
		return pi
