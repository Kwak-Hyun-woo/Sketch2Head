import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils import data
from PIL import Image
import numpy as np
import os

import random
from PIL import Image
import math

pi = torch.tensor(math.pi, dtype=torch.float32).to("cuda")

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_feature_file(filename):
    return filename == 'features.pt'

def is_angles_file(filename):
    return filename == 'camera_angle.pt'

def is_pt_file(filename):
    return any(filename.endswith(extension) for extension in ['.pt', '.PT'])

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_bezier_dataset(dir):
    points = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_pt_file(fname):
                path = os.path.join(root, fname)
                points.append(path)
                
    return points

def make_feature_dataset(dir):
    features = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_feature_file(fname):
                path = os.path.join(root, fname)
                features.append(path)
    return features

def make_camera_angle_dataset(dir):
    features = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_angles_file(fname):
                path = os.path.join(root, fname)
                features.append(path)
    return features

class FSPairedDataset(data.Dataset):
    def __init__(self, root, opt, transform=None, mode='train', target = "ws"):   # target = z, ws or tri_plane
        real_pic_root = os.path.join(root, "real_pic")
        sketch_root = os.path.join(root, "sketch")
        
        # input: sketch image, target: z, w or tri-plane of real image
        self.real_features_data = make_feature_dataset(real_pic_root)
        self.sketch_data = make_dataset(sketch_root)
        self.camera_angle_data = make_camera_angle_dataset(real_pic_root)

        self.target_feature = target
        self.len_imgs = len(self.sketch_data)

        self.mode = mode
        self.opt = opt
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.sketch_data[index]
        img = Image.open(img_path).convert('RGB')

        # dir, file split
        seed_path, file_name = os.path.split(img_path) 
        seed = os.path.basename(seed_path)

        # load target feature and camera angleedfgf
        feature_path = self.real_features_data[int(seed)]
        target_f = torch.load(feature_path)[self.target_feature]

        camera_angle_idx = int(file_name[6])
        camera_angle_path = self.camera_angle_data[int(seed)]
        camera_angle = torch.load(camera_angle_path)

        # transform_params = get_params(self.opt, img.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        
        # if self.mode != 'train':
        #     A_transform = self.transform_r
        
        img = self.transform(img)

        input_dict = {'img': img, 'target': target_f, 'angle': camera_angle[camera_angle_idx], 'path': img_path, 'index': index, 'name': file_name, 'angle_idx': camera_angle_idx ,'seed': int(seed)}
        return input_dict


    def __len__(self):
        return self.len_imgs


class BezierPointsDataset(data.Dataset):
    def __init__(self, root, opt, use_transform=True, mode='train', target = "ws", device = "cuda"):   # target = z, ws or tri_plane
        b_point_root = os.path.join(root, "stroke")
        real_pic_root = os.path.join(root, "real_pic")
        
        self.real_features_data = make_feature_dataset(real_pic_root)
        self.b_point_data = make_bezier_dataset(b_point_root)
        self.camera_angle_data = make_camera_angle_dataset(real_pic_root)

        self.target_feature = target
        self.len_data = len(self.b_point_data)

        self.mode = mode
        self.opt = opt
        self.use_transform = use_transform
        self.device = device

    def resize(self, tensor, scale):
        resize_matrix = torch.tensor([[scale, 0],
                                    [0, scale]]).to(torch.float32).to(self.device)
        return torch.matmul(tensor, resize_matrix)
    
    def flip(self, tensor, axis='x'):
        if axis == 'x':
            flip_matrix = torch.tensor([[1, 0],
                                        [0, -1]]).to(torch.float32).to(self.device)
        elif axis == 'y':
            flip_matrix = torch.tensor([[-1, 0],
                                        [0, 1]]).to(torch.float32).to(self.device)
        return torch.matmul(tensor, flip_matrix)
    
    def rotate(self, tensor, angle_degrees):
        angle_radians = torch.tensor(angle_degrees, dtype=torch.float32).to(self.device).to(self.device)* (pi / 180)
        rotate_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)],
                                    [torch.sin(angle_radians), torch.cos(angle_radians)]]).to(torch.float32).to(self.device)
        rotated_points =  torch.matmul(tensor, rotate_matrix)

        # resize 
        min_point = min(rotated_points[:, 0, :].min(), rotated_points[:, 3, :].min())
        max_point = max(rotated_points[:, 0, :].max(), rotated_points[:, 3, :].max())
        if min_point < 0 or max_point > 224:
            # expand 
            return self.resize(rotated_points, 224/(max_point-min_point))
        else:
            return rotated_points
        
    def transform_func(self, tensor):
        # test 
        # return self.resize(tensor, 0.8)
        # return self.flip(tensor)
        # return self.rotate(tensor, 10)
        # return tensor
        transform_funcs = [self.resize, self.flip, self.rotate, "Pass"]
        selected_method = random.choice(transform_funcs)
        if selected_method == "Pass":
            return tensor
        if selected_method == self.resize:
            return selected_method(tensor, random.uniform(0.75, 1.0))
        if selected_method == self.flip:
            return selected_method(tensor)
        if selected_method == self.rotate:
            return selected_method(tensor, random.uniform(-20, +20))
 
    def __getitem__(self, index):
        b_point_path = self.b_point_data[index]
        b_point = torch.stack(torch.load(b_point_path))

        # dir, file split
        seed_path, file_name = os.path.split(b_point_path) 
        seed = os.path.basename(seed_path)

        # load target feature and camera angle
        feature_path = self.real_features_data[int(seed)]
        target_f = torch.load(feature_path)[self.target_feature]

        camera_angle_idx = int(file_name[6])
        camera_angle_path = self.camera_angle_data[int(seed)]
        camera_angle = torch.load(camera_angle_path)

        # transform_params = get_params(self.opt, img.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        
        # if self.mode != 'train':
        #     A_transform = self.transform_r
        if self.use_transform:
            b_point = self.transform_func(b_point)

        input_dict = {'b_point': b_point, 'target': target_f, 'angle': camera_angle[camera_angle_idx], 'path': b_point_path, 'index': index, 'name': file_name, 'angle_idx': camera_angle_idx ,'seed': int(seed)}
        return input_dict

    def __len__(self):
        return self.len_data

class ResterDataset(data.Dataset):
    def __init__(self, root, opt, transform=None, mode='train', target = "ws"):   # target = z, ws or tri_plane
        real_pic_root = os.path.join(root, "real_pic")
        rester_root = os.path.join(root, "stroke")
        
        # input: sketch image, target: z, w or tri-plane of real image
        self.real_features_data = make_feature_dataset(real_pic_root)
        self.rester_data = make_dataset(rester_root)
        self.camera_angle_data = make_camera_angle_dataset(real_pic_root)

        self.target_feature = target
        self.len_data = len(self.rester_data)

        self.mode = mode
        self.opt = opt
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.rester_data[index]
        img = Image.open(img_path).convert('RGB')

        # dir, file split
        seed_path, file_name = os.path.split(img_path) 
        seed = os.path.basename(seed_path)

        # load target feature and camera angle
        feature_path = self.real_features_data[int(seed)]
        target_f = torch.load(feature_path)[self.target_feature]

        camera_angle_idx = int(file_name[6])
        camera_angle_path = self.camera_angle_data[int(seed)]
        camera_angle = torch.load(camera_angle_path)

        # transform_params = get_params(self.opt, img.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        
        # if self.mode != 'train':
        #     A_transform = self.transform_r
        
        img = self.transform(img)

        input_dict = {'img': img, 'target': target_f, 'angle': camera_angle[camera_angle_idx], 'path': img_path, 'index': index, 'name': file_name, 'angle_idx': camera_angle_idx ,'seed': int(seed)}
        return input_dict


    def __len__(self):
        return self.len_data