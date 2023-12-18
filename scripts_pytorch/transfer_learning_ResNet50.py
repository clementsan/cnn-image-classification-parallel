# -*- coding: utf-8 -*-

"""
2-step transfer learning

"""


from __future__ import print_function, division

import torch
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import sys
import os
import copy

######################################################################
from model import *
import utils
from network import *



######################################################################
# Parameters
# ---------
path = '../script-preprocessing'
# Batch size
bs = 16
# Image size
sz = 224
# Learning rate
lr1 = 1e-4
lr2 = 1e-5
# Number Epochs
nb_epochs1 = 15 #25
nb_epochs2 = 30 #25
class_names = ['Class1', 'Class2', 'Class3', 'Class4']
# Number of classes
num_classes = 4
# Number of samples / ROIs block (for splitting)
num_img_split = 9


# --------
# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():


	plt.ion()   # interactive mode


	######################################################################
	# Load Data
	# ---------

	# Data augmentation and normalization for training
	# Just normalization for validation
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomRotation(5),
			transforms.ColorJitter(),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomResizedCrop(sz),
			transforms.ToTensor(),
			normalize
		]),
		'val': transforms.Compose([
			#transforms.Resize(256),
			transforms.CenterCrop(sz),
			transforms.ToTensor(),
			normalize
		]),
	}

	# ---------
	data_dict = {}
	dataloaders_dict = {}	

	for x in ['train', 'val']:
		since = time.time()
		data_list = os.path.join( path, (x + '_All.csv'))
		print(data_list)
		data = utils.load_data(data_list, data_transforms[x])

		data_dict[x] = data
		dataloaders_dict[x] = torch.utils.data.DataLoader(data_dict[x], batch_size=bs, shuffle=True, num_workers=1)

		time_elapsed = time.time() - since
		print('--- Finish loading ' + x + ' data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))
		

	# ----------------------
	# Visualize input data

	# Get a batch of training data
	inputs1, inputs2, inputs3, inputs4, classes = next(iter(dataloaders_dict['train']))
	#print(inputs1)
	#print(classes)
	#print(inputs1.type())

	# Make a grid from batch
	out = torchvision.utils.make_grid(inputs1)

	utils.imshow(out, title=[class_names[x] for x in classes])


	######################################################################
	# Transfer learning - step 1: fixed features 
	# ----------------------
	#
	# Load a pretrained model and reset final fully connected layer.
	#

	# ----------------------
	# Create model
	model_ft = Model(num_classes,num_img_split)


	# ----------------------
	# Train and evaluate
	print("\n")
	print('-' * 20)
	print("Transfer learning - Step1...")
	
	TransferLearningStep = "Step1"
	model_ft.train_model(TransferLearningStep, dataloaders=dataloaders_dict, lr=lr1, nb_epochs=nb_epochs1)
	
	# ----------------------
	# Evaluate on validation data
	model_ft.test_model(dataloaders_dict, class_names)
	
	
	######################################################################
	# Transfer learning - step 2: Finetuning the convnet
	# ----------------------
	#
	
	# ----------------------
	# Train and evaluate
	print("\n")
	print('-' * 20)
	print("Fine tuning...")
	
	TransferLearningStep = "Step2"
	model_ft.train_model(TransferLearningStep, dataloaders=dataloaders_dict, lr=lr2, nb_epochs=nb_epochs2)
	
	
	# ----------------------
	# Evaluate on validation data
	model_ft.test_model(dataloaders_dict, class_names)


	# ----------------------
	# Display predicted images
	#visualize_model(model_ft, dataloaders_dict, class_names)

	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()
