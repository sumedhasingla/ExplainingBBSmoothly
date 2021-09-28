import os
import pickle
import numpy as np
from tqdm import tqdm
import scipy.misc as scm
import pdb
import os
#import cv2
from glob import glob
from collections import namedtuple

def crop_center(img,cropx,cropy):
    
    if len(img.shape) == 3:
        img = img[:,:,0]
    try:
        y,x = img.shape
    except:
        pdb.set_trace()
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def read_data_file(file_path, image_dir=''):
    attr_list = {}   
    path = file_path
    file = open(path,'r')    
    n = file.readline()
    n = int(n.split('\n')[0]) #  Number of images    
    attr_line = file.readline()
    attr_names = attr_line.split('\n')[0].split() # attribute name
    for line in file:
        row = line.split('\n')[0].split()
        img_name = os.path.join(image_dir, row.pop(0))
        try:
            row = [float(val) for val in row]
        except:
            print(line)
            img_name = img_name + ' '+row[0]
            row.pop(0)
            row = [float(val) for val in row]            
#    img = img[..., ::-1] # bgr to rgb
        attr_list[img_name] = row       
    file.close()
    return attr_names, attr_list

def load_images_and_labels(imgs_names, image_dir, n_class, attr_list, input_size=128, num_channel=3, do_center_crop=False, uncertain = 0, crop_size = 450, read_mask=False):
    imgs = np.zeros((imgs_names.shape[0], input_size, input_size, num_channel), dtype=np.float32)
    labels = np.zeros((imgs_names.shape[0], n_class), dtype=np.float32)
    if read_mask:
        masks = np.zeros((imgs_names.shape[0], input_size, input_size,num_channel), dtype=np.float32)
    for i, img_name in tqdm(enumerate(imgs_names)):
        try:
            img = scm.imread(os.path.join(image_dir, img_name))
            if read_mask:
                mask_file_name = img_name.split('.')[0]
                mask_file_name = mask_file_name+'_256_256_SD_mask.npy'
                mask = np.load(os.path.join(image_dir, mask_file_name))
        except:
            print("Error in reading image: ", img_name)
            if i == 0:
                a=1
            else:
                a=i-1
            img = scm.imread(os.path.join(image_dir, imgs_names[a]))
            img_name = imgs_names[a]
            if read_mask:
                mask_file_name = img_name.split('.')[0]
                mask_file_name = mask_file_name+'_256_256_SD_mask.npy'
                mask = np.load(os.path.join(image_dir, mask_file_name))
                
                
        if do_center_crop and input_size == 128:
            img = crop_center(img, 150,150)
        elif do_center_crop and input_size == 256:
            img = crop_center(img, crop_size,crop_size)
            if read_mask:
                mask = crop_center(mask, 200,200)
                #mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    
        img = scm.imresize(img, [input_size, input_size, num_channel])
        img = np.reshape(img, [input_size,input_size,num_channel])
        img = img / 255.0 
        img = img - 0.5
        img = img * 2.0
        imgs[i] = img  
        if read_mask:
            mask[mask>0]=1
            mask[mask!=1]=0
            mask = np.reshape(mask, [input_size,input_size,num_channel])
            masks[i] = mask
        try:
            labels[i] = attr_list[img_name]
        except:
            print(img_name)
            pdb.set_trace()
    
    labels[np.where(labels==-1)] = uncertain
    if read_mask:
        return imgs, labels, masks
    return imgs, labels

def load_images(imgs_names, image_dir, input_size=128, num_channel=3, do_center_crop=False, crop_size=450, read_mask=False):
    imgs = np.zeros((imgs_names.shape[0], input_size, input_size, num_channel), dtype=np.float32)
    if read_mask:
        masks = np.zeros((imgs_names.shape[0], input_size, input_size,num_channel), dtype=np.float32)
    for i, img_name in tqdm(enumerate(imgs_names)):
        try:
            img = scm.imread(os.path.join(image_dir, img_name))
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.mean(img,axis=2)
            if read_mask:
                mask_file_name = img_name.split('.')[0]
                mask_file_name = mask_file_name+'_256_256_SD_mask.npy'
                mask = np.load(os.path.join(image_dir, mask_file_name))
        except:
            print("Error in reading image: ", img_name)
            if i == 0:
                a=1
            else:
                a=i-1
            img = scm.imread(os.path.join(image_dir, imgs_names[a]))
            img_name = imgs_names[a]
            if read_mask:
                mask_file_name = img_name.split('.')[0]
                mask_file_name = mask_file_name+'_256_256_SD_mask.npy'
                mask = np.load(os.path.join(image_dir, mask_file_name))
        
        if do_center_crop and input_size == 128:
            img = crop_center(img, 150,150)
        elif do_center_crop and input_size == 256:
            img = crop_center(img, crop_size,crop_size)
            if read_mask:
                mask = crop_center(mask, 200,200)
                #mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    
        img = scm.imresize(img, [input_size, input_size, num_channel])
        img = np.reshape(img, [input_size,input_size,num_channel])
        img = img / 255.0 
        img = img - 0.5
        img = img * 2.0
        imgs[i] = img
        if read_mask:
            mask[mask>0]=1
            mask[mask!=1]=0
            mask = np.reshape(mask, [input_size,input_size,num_channel])
            masks[i] = mask
    if read_mask:
        return imgs, masks
    return imgs

def load_nump(imgs_names, image_dir, input_size=128, num_channel=3, do_center_crop=False, crop_size=400):
    imgs = np.zeros((imgs_names.shape[0], input_size, input_size, num_channel), dtype=np.float32)
    for i, img_name in tqdm(enumerate(imgs_names)):
        img = np.load(os.path.join(image_dir, img_name))
        img = scm.toimage(img)
        img = scm.imresize(img, [2*input_size, 2*input_size,num_channel])
        img = crop_center(img, crop_size,crop_size)
        img = scm.imresize(img, [input_size, input_size, num_channel])
        img = np.reshape(img, [input_size,input_size,num_channel])
        img[img < int(255/2)]=0
        img[img != 0]=1
        imgs[i] = img
        
    return imgs
    


def inverse_image(img): 
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)
    
def make3d(img, num_channel, image_size, row, col):
    # img.shape = [row*col, h, w, c]
    # final: [row*h, col*w, c]
    if num_channel > 1:
        img = np.reshape(img, [col,row,image_size,image_size,num_channel]) # [col, row, h, w, c]
    else:
        img = np.reshape(img, [col,row,image_size,image_size]) # [col, row, h, w]
    img = unstack(img, axis=0) # col * [row, h, w, c]
    img = np.concatenate(img, axis=2) # [row, h, col*w, c]
    img = unstack(img, axis=0) # row * [h, col*w, c]
    img = np.concatenate(img, axis=0) # [row*h, col*w, c]
    return img


def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr
    
def save_images(realA, realB, fakeB, cycA, sample_file, image_size=128, num_channel = 3):
    img = np.concatenate((realA[:5,:,:,:],fakeB[:5,:,:,:], cycA[:5,:,:,:], realB[:5,:,:,:],
                          realA[5:,:,:,:],fakeB[5:,:,:,:], cycA[5:,:,:,:], realB[5:,:,:,:]), axis=0)
                          
    img = make3d(img, num_channel=num_channel, image_size=image_size, row=5, col=8)                      
    img = inverse_image(img)
    scm.imsave(sample_file, img)
    
def process_images_NLM_MontgomeryCXRSet(data_dir, file_name,show=False):
    try:
        img = plt.imread(os.path.join(data_dir, 'CXR_png',file_name))
    except:
        return
    mask_l = plt.imread(os.path.join(data_dir, 'ManualMask/leftMask',file_name))
    mask_r = plt.imread(os.path.join(data_dir, 'ManualMask/rightMask',file_name))
    mask = mask_l + mask_r
    mask[mask>0.5]=1
    mask[mask!=1]=0
    
    plt.imshow(img)
    plt.title(img.shape)
    if show:
        plt.show()
    else:
        plt.close()
    row_count = []
    for i in range(img.shape[0]):
        row_count.append(np.unique(img[i,:]).shape[0])
    row_count = np.asarray(row_count)
    col_count = []
    for i in range(img.shape[1]):
        col_count.append(np.unique(img[:,i]).shape[0])
    col_count = np.asarray(col_count)
    index = np.where(row_count > 10)
    min_row = index[0][0]
    max_row = index[0][-1]
    index = np.where(col_count > 10)
    min_col = index[0][0]
    max_col = index[0][-1]
    new_img = img[ min_row:max_row,min_col:max_col]
    new_mask = mask[ min_row:max_row,min_col:max_col]
    plt.imshow(new_img)
    plt.title(new_img.shape)
    if show:
        plt.show()
    else:
        plt.close()
    h,w = new_img.shape
    min1 = min(h,w)
    diff = np.abs(h - w)
    if diff > 800:
        diff = diff - 800
        if h < w:
            new_img1 = np.zeros([h+diff, w])
            new_mask1 = np.zeros([h+diff, w])
            n = int(diff/2)
            new_img1[n:n+h,:] = new_img
            new_mask1[n:n+h,:] = new_mask
        else:
            new_img1 = np.zeros([h, w+diff])
            new_mask1 = np.zeros([h, w+diff])
            n = int(diff/2)
            new_img1[:,n:n+h] = new_img
            new_mask1[:,n:n+h] = new_mask
    else:
        new_img1=new_img
        new_mask1 = new_mask
    plt.imshow(new_img1)
    plt.title(new_img1.shape)
    if show:
        plt.show()
    else:
        plt.close()
    h,w = new_img1.shape
    min1 = min(h,w)
    img1 = crop_center(new_img1,cropx=min1,cropy=min1)
    mask1 = crop_center(new_mask1,cropx=min1,cropy=min1)
    img2 = np.interp(img1, [np.min(img1),np.max(img1)], [0,255])
    mask2 = np.interp(mask1, [0,1], [0,255])
    plt.imshow(img2*mask1)
    plt.title(img2.shape)
    plt.show()
    cv.imwrite(os.path.join(data_dir, 'CXR_png_processed',file_name),img2)
    cv.imwrite(os.path.join(data_dir, 'CXR_png_processed_Masks',file_name),mask2)
    return 