from tqdm import tqdm
import numpy as np
import torch
from extract_patches_VS import *
from extract_patches_OD import *
from dataset import TestDataset
from torch.utils.data import DataLoader

class Test_Vessel_Segmentation():
	def __init__(self, args, test_img_path):
		self.args = args
		self.test_img_path = test_img_path
		assert (args.stride_height_VS <= args.test_patch_height_VS and args.stride_width_VS <= args.test_patch_width_VS)

		#Extract Patches
		self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_data_test_overlap_VS(
			test_img_path=test_img_path,
			patch_height=args.test_patch_height_VS,
			patch_width=args.test_patch_width_VS,
			stride_height=args.stride_height_VS,
			stride_width=args.stride_width_VS
			)
		self.img_height =self.test_imgs.shape[2]
		self.img_width =self.test_imgs.shape[3]

		test_set = TestDataset(self.patches_imgs_test)
		self.test_loader = DataLoader(test_set, batch_size=args.batch_size_VS, shuffle=False, num_workers=0)

	def inference(self, net):
		net.eval()
		preds_outputs = []
		preds_prob_dist = []
		with torch.no_grad():
			for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
				#inputs = inputs.cuda() #if you are using GPU
				outputs = net(inputs)
				outputs_prob_dist = outputs[:,1].data.cpu().numpy() #probability distribution
				outputs_mask = outputs.argmax(dim = 1).data.numpy() #segmentation mask
				preds_prob_dist.append(outputs_prob_dist)
				preds_outputs.append(outputs_mask)
		predictions_mask = np.concatenate(preds_outputs, axis=0)
		predictions_prob_dist = np.concatenate(preds_prob_dist, axis=0)
		self.pred_patches_mask = np.expand_dims(predictions_mask,axis=1)
		self.pred_patches_prob = np.expand_dims(predictions_prob_dist,axis=1)

		return self.pred_patches_mask, self.pred_patches_prob

	def evaluate(self):
		self.pred_imgs_mask = recompone_overlap_VS(
			self.pred_patches_mask, self.new_height, self.new_width, self.args.stride_height_VS, self.args.stride_width_VS)

		self.pred_imgs_mask = self.pred_imgs_mask[:, :, 0:self.img_height, 0:self.img_width]

		self.pred_imgs_prob_dist = recompone_overlap_VS(
			self.pred_patches_prob, self.new_height, self.new_width, self.args.stride_height_VS, self.args.stride_width_VS)

		self.pred_imgs_prob_dist = self.pred_imgs_prob_dist[:, :, 0:self.img_height, 0:self.img_width]

		return self.pred_imgs_mask, self.pred_imgs_prob_dist


class Test_OD_Segmentation():
	def __init__(self, args, test_img):
		self.args = args
		self.test_img = test_img
		assert (args.stride_height_OD <= args.test_patch_height_OD and args.stride_width_OD <= args.test_patch_width_OD)

		#Extract Patches
		self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_data_test_overlap_OD(
			test_img=test_img,
			patch_height=args.test_patch_height_OD,
			patch_width=args.test_patch_width_OD,
			stride_height=args.stride_height_OD,
			stride_width=args.stride_width_OD
			)
		self.img_height =self.test_imgs.shape[2]
		self.img_width =self.test_imgs.shape[3]

		test_set = TestDataset(self.patches_imgs_test)
		self.test_loader = DataLoader(test_set, batch_size=args.batch_size_OD, shuffle=False, num_workers=0)

	def inference(self, net):
		net.eval()
		preds_outputs = []
		preds_prob_dist = []
		with torch.no_grad():
			for batch_idx, inputs in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
				#inputs = inputs.cuda() #if you are using GPU
				outputs = net(inputs)
				outputs_prob_dist = outputs[:,1].data.cpu().numpy() #probability distribution
				outputs_mask = outputs.argmax(dim = 1).data.numpy() #segmentation mask
				preds_prob_dist.append(outputs_prob_dist)
				preds_outputs.append(outputs_mask)
		predictions_mask = np.concatenate(preds_outputs, axis=0)
		predictions_prob_dist = np.concatenate(preds_prob_dist, axis=0)
		self.pred_patches_mask = np.expand_dims(predictions_mask, axis=1)
		self.pred_patches_prob = np.expand_dims(predictions_prob_dist, axis=1)

		return self.pred_patches_mask, self.pred_patches_prob

	def evaluate(self):
		self.pred_imgs_mask = recompone_overlap_OD(
			self.pred_patches_mask, self.new_height, self.new_width, self.args.stride_height_OD, self.args.stride_width_OD)

		self.pred_imgs_mask = self.pred_imgs_mask[:, :, 0:self.img_height, 0:self.img_width]

		self.pred_imgs_prob_dist = recompone_overlap_OD(
			self.pred_patches_prob, self.new_height, self.new_width, self.args.stride_height_OD, self.args.stride_width_OD)

		self.pred_imgs_prob_dist = self.pred_imgs_prob_dist[:, :, 0:self.img_height, 0:self.img_width]

		return self.pred_imgs_mask, self.pred_imgs_prob_dist


