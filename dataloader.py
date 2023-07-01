from torch.utils import data
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
import numpy as np
from scipy import io
import random
import os


class SaliconDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png', val=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.val = val
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + '.mat')

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256, 256))

        fixation = io.loadmat(fix_path)['im']
        fixation = np.array(fixation)
        fixation = fixation.astype('float')
        if np.max(fixation) > 1.0:
            fixation = fixation / 255.0

        fixation = self.resize_fixation(fixation, 256, 256)

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0
        assert np.min(fixation) == 0.0 and np.max(fixation) == 1.0

        return img, torch.FloatTensor(gt), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.img_ids)
    def resize_fixation(self, image, row, col):
        resized_fixation = np.zeros((row, col))
        ratio_row = row / image.shape[0]
        ratio_col = col / image.shape[1]

        coords = np.argwhere(image)
        for coord in coords:
            coord_r = int(np.round(coord[0]*ratio_row))
            coord_c = int(np.round(coord[1]*ratio_col))
            if coord_r == row:
                coord_r -= 1
            if coord_c == col:
                coord_c -= 1
            resized_fixation[coord_r, coord_c] = 1

        return resized_fixation


class TestLoader(DataLoader):
    def __init__(self, img_dir, img_ids):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        sz = img.size
        img = self.img_transform(img)
        return img, img_id, sz

    def __len__(self):
        return len(self.img_ids)


class MITDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.jpg', val=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.val = val
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpeg')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256, 256))

        fixations = np.array(Image.open(fix_path).convert('L'))
        fixations = fixations.astype('float')

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        fixations = (fixations > 0.5).astype('float')

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0
        assert np.min(fixations) == 0.0 and np.max(fixations) == 1.0

        return img, torch.FloatTensor(gt), torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.img_ids)

class S4ASDDataset(DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png', val=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.val = val
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.png')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + '.mat')

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256, 256))

        fixation = io.loadmat(fix_path)['im']
        fixation = np.array(fixation)
        fixation = fixation.astype('float')
        if np.max(fixation) > 1.0:
            fixation = fixation / 255.0

        fixation = self.resize_fixation(fixation, 256, 256)

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0
        assert np.min(fixation) == 0.0 and np.max(fixation) == 1.0

        return img, torch.FloatTensor(gt), torch.FloatTensor(fixation)

    def __len__(self):
        return len(self.img_ids)
    def resize_fixation(self, image, row, col):
        resized_fixation = np.zeros((row, col))
        ratio_row = row / image.shape[0]
        ratio_col = col / image.shape[1]

        coords = np.argwhere(image)
        for coord in coords:
            coord_r = int(np.round(coord[0]*ratio_row))
            coord_c = int(np.round(coord[1]*ratio_col))
            if coord_r == row:
                coord_r -= 1
            if coord_c == col:
                coord_c -= 1
            resized_fixation[coord_r, coord_c] = 1

        return resized_fixation



class MyDataset(data.Dataset):
    def __init__(self, datalist_image, datalist_label, datalist_fixation, fixPts_format):
        super(MyDataset, self).__init__()
        self.datalist_image = datalist_image
        self.datalist_label = datalist_label
        self.datalist_fixation = datalist_fixation
        self.fixPts_format = fixPts_format

        self.files_image = []
        self.files_label = []
        self.files_fixation = []
        for file in open(self.datalist_image):
            self.files_image.append(file.split('\n')[0])
        for file in open(self.datalist_label):
            self.files_label.append(file.split('\n')[0])
        if self.datalist_fixation is not None:
            for file in open(self.datalist_fixation):
                self.files_fixation.append(file.split('\n')[0])

    def __getitem__(self, index):
        image = Image.open(self.files_image[index]).convert('RGB')
        label = Image.open(self.files_label[index]).convert('L')
        if self.fixPts_format == 'salicon':
            fixation = io.loadmat(self.files_fixation[index])['im']
            fixation = np.array(fixation)
            fixation_fine = self.resize_fixation(fixation, 256, 256)
            other_map = io.loadmat(self.files_fixation[random.randint(0, 4999)])["im"]
            other_map = self.resize_fixation(other_map, 256, 256)
        elif self.fixPts_format == 'mit':
            fixation = Image.open(self.files_fixation[index]).convert('L')
            fixation = np.array(fixation)
            fixation_fine = self.resize_fixation(fixation, 256, 256)
            other_map = Image.open(self.files_fixation[random.randint(0, 99)]).convert('L')
            other_map = np.array(other_map)
            other_map = self.resize_fixation(other_map, 256, 256)
        elif self.fixPts_format == 's4asd':
            fixation = io.loadmat(self.files_fixation[index])['im']
            fixation = np.array(fixation)

            fixation = fixation.astype('float')
            if np.max(fixation) > 1.0:
                fixation = fixation / 255.0


            fixation_fine = self.resize_fixation(fixation, 256, 256)
            other_map = io.loadmat(self.files_fixation[random.randint(0, 29)])["im"]
            other_map = self.resize_fixation(other_map, 256, 256)

        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])
        image = trans(image)

        label = np.array(label)
        label = label.astype('float')
        label_fine = cv2.resize(label, (256, 256))
        label_fine = label_fine / 255.0

        return image, torch.FloatTensor(label_fine), torch.FloatTensor(fixation_fine)

    def __len__(self):
        return len(self.files_image)

    def resize_fixation(self, image, row, col):
        resized_fixation = np.zeros((row, col))
        ratio_row = row / image.shape[0]
        ratio_col = col / image.shape[1]

        coords = np.argwhere(image)
        for coord in coords:
            coord_r = int(np.round(coord[0]*ratio_row))
            coord_c = int(np.round(coord[1]*ratio_col))
            if coord_r == row:
                coord_r -= 1
            if coord_c == col:
                coord_c -= 1
            resized_fixation[coord_r, coord_c] = 1

        return resized_fixation