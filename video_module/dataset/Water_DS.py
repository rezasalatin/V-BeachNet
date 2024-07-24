import os
import numpy as np

from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from video_module.dataset import transforms as mytrans
import myutils


class Water_Image_Train_DS(data.Dataset):

    def __init__(self, root, output_size, clip_n, max_obj_n):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        self.img_list = sorted(glob(os.path.join(root, 'JPEGImages', '*.jpg')) + glob(os.path.join(root, 'JPEGImages', '*.png')))
        self.mask_list = sorted(glob(os.path.join(root, 'Annotations', '*.png')))

        assert len(self.img_list) == len(self.mask_list), "The number of images and masks should be the same"

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = mytrans.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_pil = myutils.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = myutils.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        for i in range(self.clip_n):
            img, mask = img_pil, mask_pil
            if i > 0:
                img, mask = self.random_horizontal_flip(img, mask)
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            img, mask = self.random_resize_crop(img, mask)
            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            frames[i] = self.to_tensor(img)
            masks[i] = mask

        info = {
            'name': self.img_list[idx]
        }
        return frames, masks[:, :obj_n], obj_n, info



class Video_DS(data.Dataset):

    def __init__(self, img_list, first_frame, first_mask):
        self.img_list = img_list[1:]
        self.video_len = len(self.img_list)

        first_mask = np.array(first_mask, np.uint8) > 0
        self.obj_n = first_mask.max() + 1

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(self.obj_n, shuffle=False)

        mask, _ = self.to_onehot(first_mask)
        self.first_mask = mask[:self.obj_n]
        self.first_frame = self.to_tensor(first_frame)

    def __len__(self):
        return self.video_len

    def __getitem__(self, idx):
        img = myutils.load_image_in_PIL(self.img_list[idx], 'RGB')
        frame = self.to_tensor(img)
        img_name = os.path.basename(self.img_list[idx])[:-4]

        return frame, img_name
