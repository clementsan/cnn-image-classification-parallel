# -*- coding: utf-8 -*-

"""
2-step transfer learning

"""


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import sys
import os
import copy

#from torchsummary import summary
from sklearn.metrics import confusion_matrix

######################################################################
#from os import path
#sys.path.append( path.dirname( path.abspath(__file__) ) )
#from .layers import *
from layers import *

######################################################################
# Parameters
# ---------
path = '~/Projects/Project_SEM/Project_TargetClass/data'
# Batch size
bs = 32
# Image size
sz = 224
# Learning rate
lr1 = 1e-3
lr2 = 1e-3
# Number Epochs
nb_epochs1 = 25
nb_epochs2 = 25

# --------
# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



######################################################################
# Visualize a few images


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



######################################################################
# Training the model
# ------------------



def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Testing model predictions

def test_model(model, dataloaders, class_names):
    print("prediction on validation data")
    model.load_state_dict(torch.load('pytorch_model.h5'))
    model.eval()
    total_labels = []
    total_preds = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            #print("DataLoader iteration: %d" % i)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #print("\t Predictions: %s" % preds.data.cpu().numpy())
            #print("\t Labels: %s" % labels.data.cpu().numpy())
            total_labels.extend(labels.data.cpu().numpy())
            total_preds.extend(preds.data.cpu().numpy())

    print(class_names)
    cm = confusion_matrix(total_labels,total_preds)
    print(cm)

######################################################################
# Main function

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
            transforms.Resize(256),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            normalize
        ]),
    }


    # ---------
    data_dir = path
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[class_names[x] for x in classes])


    ######################################################################
    # Transfer learning - step 1: fixed features 
    # ----------------------
    #
    # Load a pretrained model and reset final fully connected layer.
    #

    # Use ResNet50 model
    model_ft = models.resnet50(pretrained=True)

    # Freeze layers
    for param in model_ft.parameters():
        param.requires_grad = False

    # ------------
    # Update model
    # Single linear features
    num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 2)
    # new dense model
    model_ft.fc = nn.Sequential(
        # Note: Adaptive Concatenation generates 4096 features (both max & avg pooling)
        #AdaptiveConcatPool2d(),
        #nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        #Flatten(),
        #nn.BatchNorm1D(2048, eps=1e-05, momentum=0.1, affine=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        #nn.BatchNorm1D(512, eps=1e-05, momentum=0.1, affine=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=2)
        )
    print(model_ft)
    #print(summary(model_ft, (3, 224, 224)))

    # Dataset Overview
    print("\nSize Training dataset: %s" % dataset_sizes['train'])
    print("Size Validation dataset: %s" % dataset_sizes['val'])

    # Attach to device
    model_ft = model_ft.to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.

    optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=lr1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # ----------------------
    # Train and evaluate
    print("\n")
    print('-' * 20)
    print("Transfer learning...")
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=nb_epochs1)
    # Save trained model
    torch.save(model_ft.state_dict(),'pytorch_model.h5')
    # ----------------------
    # Evaluate on validation data
    test_model(model_ft,dataloaders, class_names)


    ######################################################################
    # Transfer learning - step 2: Finetuning the convnet
    # ----------------------
    #

    # Unfreeze layers
    for param in model_ft.parameters():
        param.requires_grad = True

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr2, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # ----------------------
    # Train and evaluate
    print("\n")
    print('-' * 20)
    print("Fine tuning...")
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=nb_epochs2)

    # Save trained model
    torch.save(model_ft.state_dict(),'pytorch_model.h5')

    # ----------------------
    # Evaluate on validation data
    test_model(model_ft,dataloaders, class_names)


    #visualize_model(model_ft, dataloaders, class_names)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
