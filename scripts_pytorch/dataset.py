from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage
from torch.utils.data import Dataset
import pandas as pd

class MyData(Dataset):
	def __init__(self, file_list):
		self.label_name = ['Class1', 'Class2', 'Class3', 'Class4']
		
		df = pd.read_csv(file_list, sep=',')
		im1_list = df['10000x'].tolist()
		im2_list = df['25000x'].tolist()
		im3_list = df['50000x'].tolist()
		im4_list = df['100000x'].tolist()
		labels_list = df['label'].tolist()
		labels_list = [self.label_name.index(label) for label in labels_list]
		num_file = df.shape[0]
		
		self.im1_list = im1_list
		self.im2_list = im2_list
		self.im3_list = im3_list
		self.im4_list = im4_list
		self.labels_list = labels_list
		self.num_file = num_file
		

	def __getitem__(self, idx):
		
		img1_name = self.im1_list[idx]
		img2_name = self.im2_list[idx]
		img3_name = self.im3_list[idx]
		img4_name = self.im4_list[idx]
		label = self.labels_list[idx]
	
		img1 = PILimage.open(img1_name, 'r').convert('RGB')
		img2 = PILimage.open(img2_name, 'r').convert('RGB')
		img3 = PILimage.open(img3_name, 'r').convert('RGB')
		img4 = PILimage.open(img4_name, 'r').convert('RGB')

		#img1 = self.gray2rgb(img1_name)
		#img2 = self.gray2rgb(img2_name)
		#img3 = self.gray2rgb(img3_name)
		#img4 = self.gray2rgb(img4_name)
		
		return img1, img2, img3, img4, label

	def gray2rgb(self, inp):

		gray = PILimage.open(inp, 'r')
		rgb = PILimage.new('RGB', gray.size)
		rgb.paste(gray)

		return rgb
