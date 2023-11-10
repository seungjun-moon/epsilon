from skimage.io import imread
from skimage.transform import rescale, resize
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import torch
from metric import mse, psnr, ssim
import os

# def SSIM(original, compressed):
#     ssim = compare_ssim(original, compressed, multichannel=True, data_range=255)
#     return ssim

def calculate_metric(path,metric):
	result_list=[]
	for file in os.listdir(path):
		if 'png' not in file and 'jpeg' not in file and 'jpg' not in file:
			continue
		image = imread(os.path.join(path,file))/ 255.
		# print(image.shape)
		size = image.shape[0]
		image_gt = image[:,0:size]
		image_out= image[:,size:size*2]

		image_gt = torch_transform(image_gt,size)
		image_out = torch_transform(image_out, size)

		### make mask
		image_sum = torch.sum(image_out, dim=1) #1,512,512?

		image_mask = torch.zeros(image_sum.shape)

		image_mask = torch.where(image_sum > 255*3-1, 0, 1)

		if metric=='psnr':
			result_list.append(psnr(image_out,image_gt, image_mask).item())

		elif metric=='ssim':
			print(ssim(image_out,image_gt))

	print('{} : {}'.format(path, sum(result_list)/len(result_list)))

def torch_transform(image, size):
	image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)

	return image

calculate_metric('./female4_epsilon','psnr')
calculate_metric('./female4_epsilon_c','psnr')
calculate_metric('./female4_epsilon_d','psnr')
calculate_metric('./female4_epsilon_e','psnr')
calculate_metric('./female4_epsilon_f','psnr')
calculate_metric('./female4_epsilon_h','psnr')
calculate_metric('./female4_epsilon_h_2','psnr')
calculate_metric('./female4_epsilon_i','psnr')
calculate_metric('./female4_epsilon','psnr')
