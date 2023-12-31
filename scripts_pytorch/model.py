from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np 
import scipy
import sys
import os
import copy
import time
import matplotlib.pyplot as plt
#from torchsummmary import summary

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from layers import *
from network import *

class Model(object):
	def __init__(self, num_classes, num_img_split):
		self.num_classes = num_classes	
		self.num_img_split = num_img_split

		# Criterion
		self.criterion = nn.CrossEntropyLoss()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# Create ResNet50 model
		self.create_model()

	def create_model(self):
		

		# Combined resNets with one final layer
		#self.model = MyNetwork1(self.num_classes)
		# More advanced network (hidden layers and dropout)
		self.model = MyNetworkFastAI2(self.num_classes)
		print(self.model)
		#print(summary(self.model, (2,224,224)))
		
		# Attach to device
		self.model = self.model.to(self.device)
		


	# Need to udpate: step1 vs step2
	def train_model(self, TransferLearningStep, dataloaders, lr, nb_epochs=25):
		since = time.time()

		# Update optimizer and schredule based on transferLearningStep
		if (TransferLearningStep == "Step1"):

			# Update for Network2
			params_to_update = []
			#search = ['fc','network1.fc','network2.fc','network3.fc','network4.fc']
			search = ['fc']
			for name, param in self.model.named_parameters():
				#print(name)
				if any(x in name for x in search):
				#if 'fc' in name:
					print("\tUpdate grad: %s" % name)
					param.requires_grad = True
					params_to_update.append(param)
				else:
					param.requires_grad = False

			self.params_to_update = params_to_update

			#Udpates for simpler network1 (one fc1 layer)
			#Freeze layers
			# for name, param in self.model.named_parameters():
			# 	print(name)
			# 	param.requires_grad = False
			# # Unfreeze layers for fc1
			# for name, param in self.model.fc1.named_parameters():
			# 	print (name)
			# 	param.requires_grad = True

			# optimizer = optim.SGD([{'params': self.params_to_update}], lr=lr, momentum=0.9)
			optimizer = optim.AdamW([{'params': self.params_to_update}], lr=lr, betas=(0.9, 0.999), weight_decay=0.3)
			
			# Decay LR by a factor of 0.1 every 7 epochs
			# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		elif (TransferLearningStep == "Step2"):
			# Unfreeze all layers
			for param in self.model.parameters():
				param.requires_grad = True

			# Observe that all parameters are being optimized
			# optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
			optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.3)
			
			# Decay LR by a factor of 0.1 every 7 epochs
			# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		else:
			print("ERROR transfer learning step")

		best_model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0

		train_loss = []
		val_loss = []
		train_acc = []
		val_acc = []

		for epoch in range(nb_epochs):
			print('Epoch {}/{}'.format(epoch, nb_epochs - 1))
			print('-' * 10)

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					scheduler.step()
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				running_corrects = 0

				# Iterate over data.
				for inputs1, inputs2, inputs3, inputs4, labels in dataloaders[phase]:
					inputs1 = inputs1.to(self.device)
					inputs2 = inputs2.to(self.device)
					inputs3 = inputs3.to(self.device)
					inputs4 = inputs4.to(self.device)
					labels = labels.to(self.device)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						# Provide two inputs to model
						outputs = self.model(inputs1, inputs2, inputs3, inputs4)
						_, preds = torch.max(outputs, 1)
						loss = self.criterion(outputs, labels)

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							optimizer.step()

					# statistics
					running_loss += loss.item() * inputs1.size(0)
					running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / len(dataloaders[phase].dataset) 
				epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) 

				curr_lr = optimizer.param_groups[0]['lr']

				print('{} Loss: {:.4f} Acc: {:.4f} Lr: {:.6f}'.format(
					phase, epoch_loss, epoch_acc, curr_lr))


				# Append values for plots
				if phase == 'train':
					train_loss.append(epoch_loss)
					train_acc.append(epoch_acc)
				else:
					val_loss.append(epoch_loss)
					val_acc.append(epoch_acc)

				# deep copy the model
				if phase == 'val' and epoch_acc >= best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(self.model.state_dict())
					# Save trained model
					torch.save(self.model.state_dict(),'pytorch_model.h5')

			print()

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		# Generate plots
		plt.figure(); plt.plot(range(1,nb_epochs+1),train_loss,'k', range(1,nb_epochs+1), val_loss, 'r')
		plt.legend(['Train Loss','Val Loss'])
		plt.savefig(os.getcwd()+ '/loss_' + TransferLearningStep + '.png')

		plt.figure(); plt.plot(range(1,nb_epochs+1),train_acc,'k', range(1,nb_epochs+1), val_acc, 'r')
		plt.legend(['Train Accuracy','Val Accuracy'])
		plt.savefig(os.getcwd()+ '/acc_' + TransferLearningStep + '.png')

		# load best model weights
		self.model.load_state_dict(best_model_wts)

		# Save trained model
		torch.save(self.model.state_dict(),'pytorch_model.h5')


	def visualize_model(self, dataloaders, class_names, num_images=6):
		was_training = self.model.training
		self.model.eval()
		images_so_far = 0
		fig = plt.figure()

		with torch.no_grad():
			for i, (inputs1, inputs2, inputs3, inputs4, labels) in enumerate(dataloaders['val']):
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				inputs3 = inputs3.to(self.device)
				inputs4 = inputs4.to(self.device)
				labels = labels.to(self.device)


				outputs = self.model(inputs1, inputs2, inputs3, inputs4)
				_, preds = torch.max(outputs, 1)

				for j in range(inputs1.size()[0]):
					images_so_far += 1
					ax = plt.subplot(num_images//2, 2, images_so_far)
					ax.axis('off')
					ax.set_title('predicted: {}'.format(class_names[preds[j]]))
					imshow(inputs1.cpu().data[j])

					if images_so_far == num_images:
						self.model.train(mode=was_training)
						return
			self.model.train(mode=was_training)


	def test_model(self, dataloaders, class_names):
		print("\nPrediction on validation data")
		was_training = self.model.training
		self.model.eval()
		#self.model.load_state_dict(torch.load('pytorch_model.h5'))
		#self.model.eval()
		total_labels = []
		total_preds = []
		with torch.no_grad():
			for i, (inputs1, inputs2, inputs3, inputs4, labels) in enumerate(dataloaders['val']):
				#print("DataLoader iteration: %d" % i)
				inputs1 = inputs1.to(self.device)
				inputs2 = inputs2.to(self.device)
				inputs3 = inputs3.to(self.device)
				inputs4 = inputs4.to(self.device)
				labels = labels.to(self.device)

				outputs = self.model(inputs1, inputs2, inputs3, inputs4)
				_, preds = torch.max(outputs, 1)
				#print("\t Predictions: %s" % preds.data.cpu().numpy())
				#print("\t Labels: %s" % labels.data.cpu().numpy())
				total_labels.extend(labels.data.cpu().numpy())
				total_preds.extend(preds.data.cpu().numpy())

		print(class_names)

		# Accuracy
		Accuracy = accuracy_score(total_labels, total_preds)
		print("Accuracy: %s" % Accuracy)

		# Classification report
		Report = classification_report(total_labels, total_preds, target_names=class_names,output_dict=True)
		print("Report: %s" %Report)

		# Confusion matrix
		cm = confusion_matrix(total_labels,total_preds)
		print(cm)

		# Majority voting
		majority_vote(total_preds,total_labels)


		self.model.train(mode=was_training)




	def majority_vote(self, preds, labels):

		num_per_img = self.num_img_split
		maj_vec = np.zeros((labels.shape[0]//num_per_img,))
		maj_labels = np.copy(maj_vec)
		
		for i in range(0,labels.shape[0],num_per_img):
			curr_mode,_ = scipy.stats.mode(preds[i:i+num_per_img])
			
			maj_vec[i//num_per_img] = curr_mode[0]
			maj_labels[i//num_per_img] = labels[i]

		acc = float(len(np.where(maj_vec == maj_labels)[0])) / len(maj_vec)
		print('Majority vote Acc = {:.6f}'.format(acc))


	def cov(self, attr, ftrs):
		mu_attr = np.mean(attr)
		mu_ftrs = np.mean(ftrs)

		vec = ftrs - mu_ftrs
		overall = 0.0
		for i in range(len(attr)):
			overall += np.sum((attr[i]-mu_attr) * vec)

		cov = overall / (len(attr) * len(ftrs))

		return cov


	def save_img(self, files, preds):

		p = preds.cpu().numpy()
		for i in range(len(files)):
			curr_name = os.getcwd() + '/predicted_img/' + files[i].split('/')[9][:-4] + '_pred.tif'
			curr_img = p[i,...]			
			#print(curr_img)
			scipy.misc.imsave(curr_name, curr_img)


	def plot_confusion_matrix(self, ma, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.figure()
		if normalize:
			ma = ma.astype('float') / ma.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')
		

		plt.imshow(ma, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = ma.max() / 2.
		for i, j in itertools.product(range(ma.shape[0]), range(ma.shape[1])):
			plt.text(j, i, format(ma[i, j], fmt),
		             horizontalalignment="center",
		             color="white" if ma[i, j] > thresh else "black")

		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		plt.savefig(os.getcwd() + '/' + title + '.tif')

