from torch import nn

class FConvNetModel(nn.Module):

	"""
	Define your fully convolutional network here
	"""

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, 5, 2, 1)
		self.conv2 = nn.Conv2d(32, 64, 5, 2, 1)
		self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
		self.conv4 = nn.Conv2d(128, 256, 5, 2, 1)

		self.upconv1 = nn.ConvTranspose2d(256, 128, 5, 2, 1)
		self.upconv2 = nn.ConvTranspose2d(128, 64, 5, 2, 1)
		self.upconv3 = nn.ConvTranspose2d(64, 32, 5, 2, 1)
		self.upconv4 = nn.ConvTranspose2d(32, 6, 5, 2, 1, 1)

		self.relu = nn.ReLU()

	def forward(self, x):

		c1 = self.conv1(x)
		c1 = self.relu(c1)
		c2 = self.conv2(c1)
		c2 = self.relu(c2)
		c3 = self.conv3(c2)
		c3 = self.relu(c3)
		c4 = self.conv4(c3)
		c4 = self.relu(c4)

		up1 = self.upconv1(c4)
		up1 = self.relu(up1)
		up2 = self.upconv2(up1+c3)
		up2 = self.relu(up2)
		up3 = self.upconv3(up2+c2)
		up3 = self.relu(up3)
		up4 = self.upconv4(up3+c1)


		return up4
