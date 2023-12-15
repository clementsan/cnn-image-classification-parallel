from __future__ import print_function, division

import numpy as np
import glob

def main():
	

	root_dir = '~/Project_SEM/Project_TargetClass/data/'
	
	CSV_NameType = 'All'
	TargetClass = ['Class1', 'Class2', 'Class3', 'Class4']
	mag = ['10000x','25000x','50000x','100000x']
	train_list = []
	val_list = []

	for i in range(len(TargetClass)):
		print("TargetClass: %s" % TargetClass[i])
		dir_list = []
		num_files = 1000
		for j in range(len(mag)):
			curr_dir = root_dir + TargetClass[i] + '/%s/*' %mag[j]
			curr_files = glob.glob(curr_dir)
			curr_files.sort()			
			if len(curr_files) < num_files:
				num_files = len(curr_files)
			dir_list.append(curr_files)			
		#print(len(dir_list))

		curr_files = []
		for j in range(num_files):						
			temp = dir_list[0][j] + ',' + dir_list[1][j] + ',' + dir_list[2][j] + ',' + dir_list[3][j] 
			curr_files.append(temp)

		np.random.shuffle(curr_files)

		print("\tMinimum curr_files: %d" % len(curr_files))

		# Split dataset into training and validation (20% random split)
		num_train = int(0.8*len(curr_files))
		print("\tnum_train: %d" % num_train)
		for j in range(len(curr_files)):

			if j < num_train:
				train_list.append(curr_files[j] + ',' + TargetClass[i] + '\n')
			else:		
				val_list.append(curr_files[j] + ',' + TargetClass[i] + '\n' )
		print("\tlength training - validation: %d - %d" % (num_train, len(curr_files)-num_train))
		#print(num_train, len(curr_files)-num_train)


	print("\nOverall: training - validation - total")
	print(len(train_list), len(val_list), len(train_list)+ len(val_list))
	np.random.shuffle(train_list)
	np.random.shuffle(val_list)

	FileName_train = './train_' + CSV_NameType + '.csv'
	fb = open(FileName_train,'w')
	fb.write('10000x,25000x,50000x,100000x,label\n')
	for i in range(len(train_list)):
		fb.write(train_list[i])
	fb.close()

	FileName_val = './val_' + CSV_NameType + '.csv'
	fb = open(FileName_val,'w')	
	fb.write('10000x,25000x,50000x,100000x,label\n')
	for i in range(len(val_list)):
		fb.write(val_list[i])
	fb.close()

	return



if __name__ == "__main__":
	main()
