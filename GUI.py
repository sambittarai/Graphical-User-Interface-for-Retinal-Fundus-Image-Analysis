#=================================================================================================================================================
# This is an application which takes a single retinal image at a time and displays its preprocessed image and segmentation mask in the GUI viewer.
#=================================================================================================================================================

import torch
from Network_Architectures import UNetFamily
import torch.backends.cudnn as cudnn
import os
import argparse
from config import parse_args
# from extract_patches import *
from extract_patches_VS import load_data_preprocess
import numpy as np
# from dataset import TestDataset
# from torch.utils.data import DataLoader
from tqdm import tqdm
from pre_process_1 import my_PreProc
from pre_process_2 import clahe_rgb
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from fractal_dimension import fractal_dimension
import tkinter.font as font
import cv2
from Test import Test_Vessel_Segmentation
from Test import Test_OD_Segmentation
from ONH_Detection import ONH_Region_Crop
from utils import getLargestCC
# progressbar - batch_idx

'''
class Test():
	def __init__(self, args, test_img_path):
		self.args = args
		self.test_img_path = test_img_path
		assert (args.stride_height <= args.test_patch_height and args.stride_width <= args.test_patch_width)

		#Extract Patches
		self.patches_imgs_test, self.test_imgs, self.new_height, self.new_width = get_data_test_overlap(
			test_img_path=test_img_path,
			patch_height=args.test_patch_height,
			patch_width=args.test_patch_width,
			stride_height=args.stride_height,
			stride_width=args.stride_width
			)
		self.img_height =self.test_imgs.shape[2]
		self.img_width =self.test_imgs.shape[3]

		test_set = TestDataset(self.patches_imgs_test)
		self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
		self.pred_imgs_mask = recompone_overlap(
			self.pred_patches_mask, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)

		self.pred_imgs_mask = self.pred_imgs_mask[:, :, 0:self.img_height, 0:self.img_width]

		self.pred_imgs_prob_dist = recompone_overlap(
			self.pred_patches_prob, self.new_height, self.new_width, self.args.stride_height, self.args.stride_width)

		self.pred_imgs_prob_dist = self.pred_imgs_prob_dist[:, :, 0:self.img_height, 0:self.img_width]

		return self.pred_imgs_mask, self.pred_imgs_prob_dist
'''

def run():
	torch.multiprocessing.freeze_support()

def Predict_Vessel(test_img_path, args):
	#device
	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") #device = cpu
	#Model Architecture
	net = UNetFamily.U_Net(args.in_channels, args.classes).to(device)
	cudnn.benchmark = True
	# Load checkpoint
	checkpoint = torch.load(args.Vessel_best_model_path, map_location=device)
	net.load_state_dict(checkpoint['net'])

	eval = Test_Vessel_Segmentation(args, test_img_path)
	pred_patches_mask, pred_patches_prob_dist = eval.inference(net)
	pred_img_mask, pred_img_prob_dist = eval.evaluate()

	return pred_img_mask, pred_img_prob_dist

def Predict_OD(patch, args):
	#device
	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") #device = cpu
	#Model Architecture
	net = UNetFamily.U_Net(args.in_channels, args.classes).to(device)
	cudnn.benchmark = True
	# Load checkpoint
	checkpoint = torch.load(args.OD_best_model_path, map_location=device)
	net.load_state_dict(checkpoint['net'])

	eval = Test_OD_Segmentation(args, patch)
	pred_patches_mask, pred_patches_prob_dist = eval.inference(net)
	pred_img_mask, pred_img_prob_dist = eval.evaluate()

	return pred_img_mask, pred_img_prob_dist

def Browse():
	global path
	path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("TIF File", "*.tif"), ("JPG File", ".jpg"), ("PNG File", "*.png"), ("All Files", "*.*")))
	img = Image.open(path)
	img.thumbnail((300, 300))
	img = ImageTk.PhotoImage(img)
	label_1.configure(image=img)
	label_1.image = img

def Show_Preprocess_1():
	global save_preprocess_1_image
	test_imgs = load_data_preprocess(path)
	test_imgs = my_PreProc(test_imgs)
	test_imgs = test_imgs * 255.
	test_imgs = np.squeeze(test_imgs, 0)
	test_imgs = np.squeeze(test_imgs, 0)
	test_imgs = Image.fromarray(test_imgs).convert('RGB')
	save_preprocess_1_image = test_imgs.copy()
	test_imgs.thumbnail((300, 300))
	test_imgs = ImageTk.PhotoImage(test_imgs)
	label_2.configure(image=test_imgs)
	label_2.image = test_imgs

# def show_preprocess_2():
# 	global test_imgs_preprocessed
# 	test_imgs_preprocessed = clahe_rgb(path, cliplimit=4, tilesize=16)
# 	test_imgs_preprocessed = Image.fromarray(test_imgs_preprocessed)

# 	test_imgs = clahe_rgb(path, cliplimit=4, tilesize=16)
# 	#test_imgs = test_imgs * 255.
# 	test_imgs = Image.fromarray(test_imgs)
# 	test_imgs.thumbnail((300, 300))
# 	test_imgs = ImageTk.PhotoImage(test_imgs)
# 	label_6.configure(image=test_imgs)
# 	label_6.image = test_imgs

def preprocess_2():
	test_imgs_preprocessed = clahe_rgb(path, cliplimit=4, tilesize=16)
	test_imgs_preprocessed = Image.fromarray(test_imgs_preprocessed)
	return test_imgs_preprocessed


def Overlap_1():
	'''
	Retinal Vessels are overlapped with the preprocessed retinal fundus image.
	'''
	# probability_distribution - numpy array

	global save_img_1

	img = preprocess_2()
	img = img.resize((int(img.size[0]/2), int(img.size[1]/2))) # comment it out if high resolution images are not being used
	img = np.asarray(img)

	th1 = 0.9
	th2 = 0.7
	th3 = 0.1

	probability_distribution_th1 = np.where(probability_distribution >= th1, 1, 0)
	probability_distribution_th2 = np.where((probability_distribution >= th2) & (probability_distribution < th1), 1, 0)
	probability_distribution_th3 = np.where((probability_distribution > th3) & (probability_distribution < th2), 1, 0)

	pred_th1, pred_th2, pred_th3 = probability_distribution_th1[0,0], probability_distribution_th2[0,0], probability_distribution_th3[0,0]
	pred_th1 = np.where(pred_th1 == 0, 1, 0)
	pred_th2 = np.where(pred_th2 == 0, 1, 0)
	pred_th3 = np.where(pred_th3 == 0, 1, 0)

	q = np.zeros((int(img.shape[0]), int(img.shape[1]), int(img.shape[2])))

	q[:,:,0] = img[:,:,0] * pred_th1
	q[:,:,1] = img[:,:,1] * pred_th1
	q[:,:,2] = img[:,:,2] * pred_th1

	pred_th1_new = np.where(pred_th1 == 0, 255, 0)
	q[:,:,0] = q[:,:,0] + pred_th1_new

	q[:,:,0] = q[:,:,0] * pred_th2 
	q[:,:,1] = q[:,:,1] * pred_th2
	q[:,:,2] = q[:,:,2] * pred_th2 

	pred_th2_new = np.where(pred_th2 == 0, 255, 0)
	q[:,:,1] = q[:,:,1] + pred_th2_new

	q[:,:,0] = q[:,:,0] * pred_th3
	q[:,:,1] = q[:,:,1] * pred_th3
	q[:,:,2] = q[:,:,2] * pred_th3

	pred_th3_new = np.where(pred_th3 == 0, 255, 0)
	q[:,:,0] = q[:,:,0] + pred_th3_new
	q[:,:,1] = q[:,:,1] + pred_th3_new

	im = q.astype(np.uint8)
	im = Image.fromarray(im)
	im = im.resize((int(img.shape[1]*2), int(img.shape[0]*2)))# comment it out if high resolution images are not being used

	save_img_1 = im.copy()

	im.thumbnail((300, 300))
	im = ImageTk.PhotoImage(im)

	label_3.configure(image=im)
	label_3.image = im

def Overlap_2():
	'''
	Optic Disc & Retinal Vessels overlapped with preprocessed retinal fundus image.
	'''
	global save_img_2
	Vessel_Mask = np.asarray(Vessel)[:,:,0]
	Vessel_Mask = np.where(Vessel_Mask > 0, 1, 0)
	Final_Mask = Vessel_Mask + OD_Mask
	Final_Mask = np.where(Final_Mask > 0, 1, 0)

	img = preprocess_2()
	img = np.asarray(img)

	prediction = np.where(Final_Mask == 0, 1, 0)
	q = np.zeros((img.shape))
	add = np.where(prediction == 0, 255, 0)

	for i in range(img.shape[2]):
		q[:,:,i] = img[:,:,i]*prediction
		q[:,:,i] = q[:,:,i] + add

	im = q.astype(np.uint8)
	im = Image.fromarray(im)
	save_img_2 = im.copy()

	im.thumbnail((300, 300))
	im = ImageTk.PhotoImage(im)
	label_4.configure(image=im)
	label_4.image = im

def calculate_f_d(pred_img_mask):
	pred_img_mask = pred_img_mask[0,0,:,:]
	f_d = fractal_dimension(pred_img_mask)
	return f_d

def show_f_d():
	l_3.configure(text="Fractal Dimension is: " + "{:.4f}".format(f_d), font=buttonFont)

def Show_Vessel_Segmentation(args):
	global f_d
	global probability_distribution
	global Vessel

	pred_img_mask, pred_img_prob_dist = Predict_Vessel(path, args)
	probability_distribution = pred_img_prob_dist.copy()

	pred_img_mask = np.where(pred_img_mask > 0, 1, 0)
	f_d = calculate_f_d(pred_img_mask)

	Vessel = pred_img_mask.copy()
	Vessel = (255. * Vessel[0,0,:,:]).astype(np.uint8)
	Vessel = Image.fromarray(Vessel).convert('RGB')
	Vessel = Vessel.resize((int(Vessel.size[0]*2), int(Vessel.size[1]*2)))

	pred_img_mask = (255. * pred_img_mask[0,0,:,:]).astype(np.uint8)
	pred_img_prob_dist = (255. * pred_img_prob_dist[0,0,:,:]).astype(np.uint8)

	pred_img_mask = Image.fromarray(pred_img_mask).convert('RGB')
	pred_img_prob_dist = Image.fromarray(pred_img_prob_dist).convert('RGB')

	pred_img_mask.thumbnail((300, 300))
	pred_img_prob_dist.thumbnail((300, 300))

	pred_img_mask = ImageTk.PhotoImage(pred_img_mask)
	pred_img_prob_dist = ImageTk.PhotoImage(pred_img_prob_dist)

	label_3.configure(image=pred_img_mask)
	label_3.image = pred_img_mask

	l_1.configure(text="Retinal Vessel Segmentation Completed", font=buttonFont)

def Show_OD_Segmentation(args):
	global OD_Mask
	test_img = np.asarray(Image.open(path))
	tl_pt, br_pt = ONH_Region_Crop(path)
	test_img_patch = test_img[tl_pt[1]:br_pt[1], tl_pt[0]:br_pt[0], :]
	patch = np.expand_dims(test_img_patch,0)
	patch = np.transpose(patch,(0,3,1,2))
	pred_img_mask, pred_img_prob_dist = Predict_OD(patch, args)
	pred_img_mask = np.where(pred_img_mask > 0, 1, 0)
	prediction_mask = np.zeros((test_img.shape[0], test_img.shape[1]))
	prediction_mask[tl_pt[1]:br_pt[1], tl_pt[0]:br_pt[0]] = pred_img_mask[0,0,:,:]
	LCC = getLargestCC(prediction_mask)
	OD_Mask = LCC.copy()
	LCC = (255. * LCC).astype(np.uint8)
	LCC = Image.fromarray(LCC).convert('RGB')
	LCC.thumbnail((300, 300))
	LCC = ImageTk.PhotoImage(LCC)
	label_4.configure(image=LCC)
	label_4.image = LCC
	l_2.configure(text="Optic Disc Segmentation Completed", font=buttonFont)

def Save_Preprocess_1():
	filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
	if not filename:
		return
	save_preprocess_1_image.save(filename)

def Save_Overlap_1():
	filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
	if not filename:
		return
	save_img_1.save(filename)

def Save_Overlap_2():
	filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
	if not filename:
		return
	save_img_2.save(filename)


root = Tk()
root.title("Retinal Vessel Segmentation")
root.configure(bg='#d9fffb')
root.iconbitmap("G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/GUI/icon.ico")
# root.geometry("850x500")
# root.resizable(False, False)
args = parse_args()

background_img = ImageTk.PhotoImage(
	Image.open("G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/GUI/background_image.gif").resize((300, 300)))
buttonFont = font.Font(family='Helvetica', size=9, weight='bold')

label_1 = Label(root, image=background_img)
label_1.grid(row=0, column=0, padx=10)

button1 = Button(root, text="Browse", command=Browse, font=buttonFont)
button1.grid(row=1, column=0, pady=5)

label_2 = Label(root, image=background_img)
label_2.grid(row=0, column=1, padx=10)

frame_2 = LabelFrame(root, text="", bg='#d9fffb', borderwidth = 0)
frame_2.grid(row=1, column=1, pady=5)

b_2_1 = Button(frame_2, text="Preprocess", font=buttonFont, command=Show_Preprocess_1)
b_2_1.pack(side=LEFT)
b_2_2 = Button(frame_2, text="Save", font=buttonFont, command=Save_Preprocess_1)
b_2_2.pack(side=RIGHT, padx=30)

label_3 = Label(root, image=background_img)
label_3.grid(row=0, column=2, padx=10)

frame_3 = LabelFrame(root, text="", bg='#d9fffb', borderwidth = 0)
frame_3.grid(row=1, column=2, pady=5)

b_3_1 = Button(frame_3, text="Overlap 1", font=buttonFont, command=Overlap_1)
b_3_1.pack(side=LEFT)
b_3_2 = Button(frame_3, text="Save", font=buttonFont, command=Save_Overlap_1)
b_3_2.pack(side=RIGHT, padx=30)

label_4 = Label(root, image=background_img)
label_4.grid(row=0, column=3, padx=10)

frame_4 = LabelFrame(root, text="", bg='#d9fffb', borderwidth = 0)
frame_4.grid(row=1, column=3, pady=5)

b_4_1 = Button(frame_4, text="Overlap 2", font=buttonFont, command=Overlap_2)
b_4_1.pack(side=LEFT)
b_4_2 = Button(frame_4, text="Save", font=buttonFont, command=Save_Overlap_2)
b_4_2.pack(side=RIGHT, padx=30)

frame_5 = LabelFrame(root, text="", bg='#d9fffb')
frame_5.grid(row=2, column=1, padx=10, pady=10)

b_1 = Button(frame_5, text="Predict Segmentation Mask", font=buttonFont, command=lambda: Show_Vessel_Segmentation(args))
b_1.grid(row=0, column=0, padx=10, pady=10)
b_2 = Button(frame_5, text="Predict Optic Disc Mask", font=buttonFont, command=lambda: Show_OD_Segmentation(args))
b_2.grid(row=1, column=0, pady=10)
b_3 = Button(frame_5, text="Predict Fractal Dimension", font=buttonFont, command=show_f_d)
b_3.grid(row=2, column=0, pady=10)

frame_6 = LabelFrame(root, text="", bg='#d9fffb')
frame_6.grid(row=2, column=2, padx=10, pady=10)

l_1 = Label(frame_6, text="")
l_1.grid(row=0, column=0, pady=10)
l_2 = Label(frame_6, text="")
l_2.grid(row=1, column=0, pady=10)
l_3 = Label(frame_6, text="")
l_3.grid(row=2, column=0, pady=10)

label_6 = Label(root, image=background_img)
label_6.grid(row=2, column=0, padx=10, pady=5)

# button = Button(root, text="Overlap 2", command=Overlap_2, font=buttonFont)
# button.grid(row=3, column=0, pady=5)

label_7 = Label(root, image=background_img)
label_7.grid(row=2, column=3, padx=10, pady=5)


root.mainloop()


# add a save button (next to browse button).
# problem - prediction looks little different on cpu and google colab. prob_dist looks little different in both the cases