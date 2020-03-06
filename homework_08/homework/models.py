from torch import nn
import torch.nn.functional as F
import torch

def one_hot(x, n=6):
	batch_size, h, w= x.size()
	x = (x.view(-1,h,w,1) == torch.arange(n, dtype=x.dtype, device=x.device)[None]).float() - torch.as_tensor([0.6609, 0.0045, 0.017, 0.0001, 0.0036, 0.314], dtype=torch.float, device=x.device)
	x = x.permute(0,3,1,2)
	return x


class FConvNetModel(nn.Module):

	def __init__(self):

		super().__init__()

		self.conv1 = nn.Conv2d(4, 32, 5, 2, 1)
		self.conv2 = nn.Conv2d(32, 64, 5, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
		self.conv4 = nn.Conv2d(128, 256, 5, 2, 1)

		self.upconv1 = nn.ConvTranspose2d(256, 128, 5, 2, 1)
		self.upconv2 = nn.ConvTranspose2d(128, 64, 5, 2, 1)
		self.upconv3 = nn.ConvTranspose2d(64, 32, 5, 2, 1)
		self.upconv4 = nn.ConvTranspose2d(32, 3, 5, 2, 1, 1)

		nn.init.constant_(self.upconv4.weight, 0)
		nn.init.constant_(self.upconv4.bias, 0)

		self.relu = nn.LeakyReLU(inplace = True)

	def forward(self, image, labels):

		image = nn.functional.interpolate(image, scale_factor=4, mode='area')
		labels = torch.unsqueeze(labels, 1).float()
		x = torch.cat((image, labels), 1)

		d1 = self.relu(self.conv1(x))
		d2 = self.relu(self.conv2(d1))
		d3 = self.relu(self.conv3(d2))
		d4 = self.relu(self.conv4(d3))

		u4 = self.relu(self.upconv1(d4))
		u3 = self.relu(self.upconv2(u4+d3))
		u2 = self.relu(self.upconv3(u3+d2))
		u1 = self.upconv4(u2+d1)

		return u1
