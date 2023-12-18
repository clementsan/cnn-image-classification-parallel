from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms



# Combined resnets with one fc layer
class MyNetwork1(nn.Module):
	def __init__(self):
		super(MyNetwork1, self).__init__()

		# Warning: need to initialize separate network, to get proper parameters (gradient updates in model.py)
		pretrained_network1 = models.resnet50(pretrained=True)
		pretrained_network2 = models.resnet50(pretrained=True)
		pretrained_network3 = models.resnet50(pretrained=True)
		pretrained_network4 = models.resnet50(pretrained=True)

		my_network1 = nn.Sequential(*list(pretrained_network1.children())[:-1])
		my_network2 = nn.Sequential(*list(pretrained_network2.children())[:-1])
		my_network3 = nn.Sequential(*list(pretrained_network3.children())[:-1])
		my_network4 = nn.Sequential(*list(pretrained_network4.children())[:-1])

		self.network1 = my_network1
		self.network2 = my_network2
		self.network3 = my_network3
		self.network4 = my_network4
		# Warning: need to define number of layers for number of features:  4 networks * 2048 featurs from ResNet50
		self.fc1 = nn.Linear(8192, 4)

	def forward(self, w, x, y, z):
		x1 = self.network1(w)
		x2 = self.network2(x)
		x3 = self.network3(y)
		# x4 size: [bs, 2048,1,1] with bs = batch-size
		x4 = self.network4(z)

		# x5 size: [bs, 4096,1,1] 
		x5 = torch.cat((x1,x2),1)
		# x46 size: [bs, 6144,1,1]
		x6 = torch.cat((x5,x3),1)
		# x7 size: [bs, 8192,1,1]
		x7 = torch.cat((x6,x4),1)

		# x10 size: [bs, 8192]
		x10 = x7.view(x7.size(0), -1)
		x10 = self.fc1(x10)

		return x10


# Combined resnets with multiple fc layers & dropout
class MyNetwork2(nn.Module):
	def __init__(self):
		super(MyNetwork2, self).__init__()

		# Warning: need to initialize separate network, to get proper parameters (gradient updates in model.py)
		self.network1 = models.resnet50(pretrained=True)
		self.network2 = models.resnet50(pretrained=True)
		self.network3 = models.resnet50(pretrained=True)
		self.network4 = models.resnet50(pretrained=True)
		#num_ftrs = self.network1.fc.in_features
		#self.network1.fc = nn.Linear(num_ftrs, 250)
		#self.network2.fc = nn.Linear(num_ftrs, 250)
		#self.network3.fc = nn.Linear(num_ftrs, 250)
		#self.network4.fc = nn.Linear(num_ftrs, 250)

		# Warning: need to define number of layers for number of features:  4 networks * 2048 featurs from ResNet50
		self.fc1_drop = nn.Dropout(p=0.5) # added this line
		self.fc2 = nn.Linear(4000, 512)
		self.fc2_drop = nn.Dropout(p=0.5) # added this line
		self.fc3 = nn.Linear(512, 4)


	def forward(self, w, x, y, z):
		# Add dropout and RELU to FC layer
		x1 = self.fc1_drop(F.relu(self.network1(w)))
		x2 = self.fc1_drop(F.relu(self.network2(x)))
		x3 = self.fc1_drop(F.relu(self.network3(y)))
		# x4 size: [bs, 1000,1,1] with bs = batch-size
		x4 = self.fc1_drop(F.relu(self.network4(z)))
		#print(x4.size())
		# Concatenate all feature layers
		# x5 size: [bs, 2000,1,1] 
		x5 = torch.cat((x1,x2),1)
		# x46 size: [bs, 3000,1,1]
		x6 = torch.cat((x5,x3),1)
		# x7 size: [bs, 4000,1,1]
		x7 = torch.cat((x6,x4),1)
		#print(x7.size())
		# x10 size: [bs, 4000]
		x10 = x7.view(x7.size(0), -1)
		#print(x10.size())
		x11 = F.relu(self.fc2(x10))
		x12 = self.fc2_drop(x11)
		#print(x12.size())
		x13 = self.fc3(x12)
		
		return x13

# Network adapted from Cuong - combined resnet18
class MyNetwork3(nn.Module):
	def __init__(self):
		super(MyNetwork3, self).__init__()

		self.model_class1 = models.resnet18(pretrained=True)
		num_ftrs = self.model_class1.fc.in_features				
		self.model_class1.fc = nn.Linear(num_ftrs, num_ftrs // 4)
		
		self.model_class2 = models.resnet18(pretrained=True)	
		num_ftrs = self.model_class2.fc.in_features					
		self.model_class2.fc = nn.Linear(num_ftrs, num_ftrs // 4)

		self.model_class3 = models.resnet18(pretrained=True)		
		num_ftrs = self.model_class3.fc.in_features					
		self.model_class3.fc = nn.Linear(num_ftrs, num_ftrs // 4)		

		self.model_class4 = models.resnet18(pretrained=True)		
		self.model_class4.fc = nn.Linear(num_ftrs, num_ftrs // 4)

		
		self.fc1 = nn.Linear(num_ftrs // 2, 4)

	def forward(self, x1, x2, x3, x4):
	#def forward(self, x3, x4):

		ftrs1 = F.relu(self.model_class1(x1))
		ftrs2 = F.relu(self.model_class2(x2))
		ftrs3 = F.relu(self.model_class3(x3))
		ftrs4 = F.relu(self.model_class4(x4))

		ftrs = torch.cat( (torch.cat( (torch.cat( (ftrs1, ftrs2), 1), ftrs3), 1), ftrs4), 1)		
		#ftrs = torch.cat( (torch.cat( (ftrs2, ftrs3), 1), ftrs4), 1)		
		#        ftrs = torch.cat( (ftrs3, ftrs4), 1)
		
		out = self.fc1(ftrs)
		
		return out, ftrs


# 
class MyNetworkCuong(nn.Module):
	def __init__(self):
		super(MyNetworkCuong, self).__init__()

		num_ftrs = 2048*2
		num_classes = 4
		mymodel = models.resnet50(pretrained=True)
		self.model_class1 = nn.Sequential(*list(mymodel.children())[:-2])

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1 = nn.Linear(num_ftrs, num_ftrs // 4)				
		
		self.BN1 = torch.nn.BatchNorm1d(num_ftrs)
		
		self.fc = nn.Linear(num_ftrs, num_classes)
		self.BN2 = torch.nn.BatchNorm1d(num_ftrs)
		self.drop = torch.nn.Dropout(p=0.5)

	def forward(self, x1, x2, x3, x4):	

		avg_pool1 = self.avg_pool( self.model_class1(x1) )
		max_pool1 = self.max_pool( self.model_class1(x1) )
		ftrs1 = self.BN2(torch.squeeze(torch.cat((avg_pool1,max_pool1),1)))

		avg_pool2 = self.avg_pool( self.model_class1(x2) )
		max_pool2 = self.max_pool( self.model_class1(x2) )
		ftrs2 = self.BN2(torch.squeeze(torch.cat((avg_pool2,max_pool2),1)))

		avg_pool3 = self.avg_pool( self.model_class1(x3) )
		max_pool3 = self.max_pool( self.model_class1(x3) )
		ftrs3 = self.BN2(torch.squeeze(torch.cat((avg_pool3,max_pool3),1)))

		avg_pool4 = self.avg_pool( self.model_class1(x4) )
		max_pool4 = self.max_pool( self.model_class1(x4) )
		ftrs4 = self.BN2(torch.squeeze(torch.cat((avg_pool4,max_pool4),1)))


		ftrs1 = F.relu( (self.fc1( ftrs1 ) ))
		ftrs2 = F.relu( (self.fc1( ftrs2 ) ))
		ftrs3 = F.relu( (self.fc1( ftrs3 ) ))
		ftrs4 = F.relu( (self.fc1( ftrs4 ) ))

		# Ftrs size: 4096 after concatenation
		ftrs = torch.cat( (torch.cat( (torch.cat( (ftrs1, ftrs2), 1), ftrs3), 1), ftrs4), 1)
		out = self.drop( self.fc( self.BN1 (ftrs) ) )

		return out, ftrs


# Network similar to fastAI (with shared weights)
class MyNetworkFastAI(nn.Module):
	def __init__(self):
		super(MyNetworkFastAI, self).__init__()

		num_ftrs = 4096
		num_classes = 4

		mymodel = models.resnet50(pretrained=True)
		self.model_class1 = nn.Sequential(*list(mymodel.children())[:-2])

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		# Change output to 512 features
		self.fc1 = nn.Linear(num_ftrs, 512)
		self.fc2 = nn.Linear(2048, 512)
		self.fc = nn.Linear(512, num_classes)

		self.BN1 = torch.nn.BatchNorm1d(num_ftrs)
		self.BN2 = torch.nn.BatchNorm1d(2048)
		self.BN3 = torch.nn.BatchNorm1d(512)
		
		self.drop1 = torch.nn.Dropout(p=0.25)
		self.drop2 = torch.nn.Dropout(p=0.25)
		self.drop3 = torch.nn.Dropout(p=0.5)

	def forward(self, x1, x2, x3, x4):	

		avg_pool1 = self.avg_pool( self.model_class1(x1) )
		max_pool1 = self.max_pool( self.model_class1(x1) )
		# Added dropout
		ftrs1 = self.drop1(self.BN1(torch.squeeze(torch.cat((avg_pool1,max_pool1),1))))

		avg_pool2 = self.avg_pool( self.model_class1(x2) )
		max_pool2 = self.max_pool( self.model_class1(x2) )
		ftrs2 = self.drop1(self.BN1(torch.squeeze(torch.cat((avg_pool2,max_pool2),1))))

		avg_pool3 = self.avg_pool( self.model_class1(x3) )
		max_pool3 = self.max_pool( self.model_class1(x3) )
		ftrs3 = self.drop1(self.BN1(torch.squeeze(torch.cat((avg_pool3,max_pool3),1))))

		avg_pool4 = self.avg_pool( self.model_class1(x4) )
		max_pool4 = self.max_pool( self.model_class1(x4) )
		ftrs4 = self.drop1(self.BN1(torch.squeeze(torch.cat((avg_pool4,max_pool4),1))))

		ftrs1 = F.relu( (self.fc1( ftrs1 ) ))
		ftrs2 = F.relu( (self.fc1( ftrs2 ) ))
		ftrs3 = F.relu( (self.fc1( ftrs3 ) ))
		ftrs4 = F.relu( (self.fc1( ftrs4 ) ))

		# Concatenation: ftrs size = 4096
		concat_ftrs = torch.cat( (torch.cat( (torch.cat( (ftrs1, ftrs2), 1), ftrs3), 1), ftrs4), 1)
		concat_ftrs = self.drop2(self.BN2(concat_ftrs))

		ftrs = self.drop3(self.BN3 (F.relu(self.fc2( concat_ftrs ))))

		out = self.fc(ftrs)

		return out, ftrs


# Network similar to fastAI (with shared weights)
class MyNetworkFastAI2(nn.Module):
	def __init__(self):
		super(MyNetworkFastAI2, self).__init__()

		num_ftrs = 4096
		num_classes = 4

		mymodel = models.resnet50(pretrained=True)
		self.model_class1 = nn.Sequential(*list(mymodel.children())[:-2])

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		# Change output to 512 features
		# self.fc1 = nn.Linear(num_ftrs*4, 4096)
		# self.fc2 = nn.Linear(4096, 512)
		# self.fc = nn.Linear(512, num_classes)
		
		# self.BN1 = torch.nn.BatchNorm1d(num_ftrs*4)
		# self.BN2 = torch.nn.BatchNorm1d(4096)
		# self.BN3 = torch.nn.BatchNorm1d(512)
		
		# self.drop1 = torch.nn.Dropout(p=0.25)
		# self.drop2 = torch.nn.Dropout(p=0.25)
		# self.drop3 = torch.nn.Dropout(p=0.5)

		self.BN1 = torch.nn.BatchNorm1d(num_ftrs*4)
		self.drop1 = torch.nn.Dropout(p=0.25)
		self.fc1 = nn.Linear(num_ftrs*4, 4096)
		self.BN2 = torch.nn.BatchNorm1d(4096)
		self.drop2 = torch.nn.Dropout(p=0.25)
		self.fc2 = nn.Linear(4096, 512)
		self.BN3 = torch.nn.BatchNorm1d(512)
		self.drop3 = torch.nn.Dropout(p=0.5)
		self.fc = nn.Linear(512, num_classes)


	def forward(self, x1, x2, x3, x4):	

		avg_pool1 = self.avg_pool( self.model_class1(x1) )
		max_pool1 = self.max_pool( self.model_class1(x1) )
		# Added dropout
		ftrs1 = torch.squeeze(torch.cat((avg_pool1,max_pool1),1))

		avg_pool2 = self.avg_pool( self.model_class1(x2) )
		max_pool2 = self.max_pool( self.model_class1(x2) )
		ftrs2 = torch.squeeze(torch.cat((avg_pool2,max_pool2),1))

		avg_pool3 = self.avg_pool( self.model_class1(x3) )
		max_pool3 = self.max_pool( self.model_class1(x3) )
		ftrs3 = torch.squeeze(torch.cat((avg_pool3,max_pool3),1))

		avg_pool4 = self.avg_pool( self.model_class1(x4) )
		max_pool4 = self.max_pool( self.model_class1(x4) )
		ftrs4 = torch.squeeze(torch.cat((avg_pool4,max_pool4),1))

		# Concatenation: ftrs size = 4096 * 4
		concat_ftrs = torch.cat( (torch.cat( (torch.cat( (ftrs1, ftrs2), 1), ftrs3), 1), ftrs4), 1)
		x = self.BN1(concat_ftrs)
		x = self.drop1(x)
		x = self.fc1(x) # 4096 output features
		x = F.relu(x)
		x = self.BN2(x)
		x = self.drop2(x)
		x = self.fc2(x) # 512 output features
		x = F.relu(x)
		x = self.BN3(x)
		ftrs = self.drop3(x)
		out = self.fc(ftrs) # 4 output features

		return out, ftrs

