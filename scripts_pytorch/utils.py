from __future__ import print_function, division

import numpy as np 
import matplotlib.pyplot as plt

from dataset import MyData

def load_data(data_list, data_transforms):
	dataclass = MyData(data_list)
	datasets = []

	for x in range(dataclass.num_file):		
		#datasets.append((data_transforms(dataclass[x][0]), dataclass[x][1]))
		
			#datasets.append((data_transforms(dataclass[x][0]),data_transforms(dataclass[x][1]),\
			#data_transforms(dataclass[x][2]),data_transforms(dataclass[x][3]),dataclass[x][4]))
			datasets.append((data_transforms(dataclass[x][0]),data_transforms(dataclass[x][1]),\
			 data_transforms(dataclass[x][2]),data_transforms(dataclass[x][3]),dataclass[x][4]))

	return datasets


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
