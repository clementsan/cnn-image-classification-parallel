from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# class MyNetworkA(nn.Module):
# 	def __init__(self, original_network):
# 		super(MyNetworkA, self).__init__()
# 		#self.network = models.resnet50(pretrained=True)
# 		# network without dense layer
# 		self.network_features = nn.Sequential(*list(original_network.children())[:-1])
# 		num_ftrs = original_network.fc.in_features
# 		self.network_features.fc = nn.Linear(in_features=num_ftrs, out_features=100)
# 	def forward(self, x):
# 		#x = self.fc1(x)
# 		x = self.network_features(x)
# 		x = self.fc(F.relu(x))
# 		return x
	

# class MyNetworkB(nn.Module):
# 	def __init__(self, original_network):
# 		super(MyNetworkB, self).__init__()
# 		#self.network = models.resnet50(pretrained=True)
# 		# network without dense layer
# 		self.network_features = nn.Sequential(*list(original_network.children())[:-1])
# 		num_ftrs = original_network.fc.in_features
# 		self.network_features.fc = nn.Linear(in_features=num_ftrs, out_features=100)
# 	def forward(self, x):
# 		#x = self.fc1(x)
# 		x = self.network_features(x)
# 		x = self.fc(F.relu(x))
# 		return x


# class MyEnsemble(nn.Module):
# 	def __init__(self, networkA, networkB):
# 		super(MyEnsemble, self).__init__()
# 		self.networkA = networkA
# 		self.networkB = networkB
# 		# need to include proper batchsize
# 		#num_ftrs = self.networkA.8.in_features
# 		#print("num_ftrs: %d" % num_ftrs)
# 		self.classifier = nn.Linear(in_features=200, out_features=2)
# 		#self.classifier = nn.Sigmoid()
# 	def forward(self, x1, x2):
# 		x1 = self.networkA(x1)
# 		print(x1.size())
# 		x2 = self.networkB(x2)
# 		print(x2.size())
# 		#combined = torch.cat((x1.view(x1.size(0),-1), x2.view(x2.size(0),-1)), dim=1)
# 		combined = torch.cat((x1,x2), dim=1)
# 		x = self.classifier(combined)
# 		return x


pretrained_network = models.resnet50(pretrained=True)
my_network1 = nn.Sequential(*list(pretrained_network.children())[:-1])
my_network2 = nn.Sequential(*list(pretrained_network.children())[:-1])
my_network3 = nn.Sequential(*list(pretrained_network.children())[:-1])
my_network4 = nn.Sequential(*list(pretrained_network.children())[:-1])

class MyNetwork(nn.Module):
	def __init__(self):
		super(MyNetwork, self).__init__()
		self.feature1 = my_network1
		self.feature2 = my_network2
		self.feature3 = my_network3
		self.feature4 = my_network4
		# Warning: need to define number of layers for number of features:  4 networks * 2048 featurs from ResNet50
		self.classifier = nn.Linear(8192, 4)

	def forward(self, w, x, y, z):
		x1 = self.feature1(w)
		x2 = self.feature2(x)
		x3 = self.feature3(y)
		# x4 size: [bs, 2048,1,1] with bs = batch-size
		x4 = self.feature4(z)

		# x5 size: [bs, 4096,1,1] 
		x5 = torch.cat((x1,x2),1)
		# x46 size: [bs, 6144,1,1]
		x6 = torch.cat((x5,x3),1)
		# x7 size: [bs, 8192,1,1]
		x7 = torch.cat((x6,x4),1)

		# x10 size: [bs, 8192]
		x10 = x7.view(x7.size(0), -1)
		x10 = self.classifier(x10)
		return x10
