# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys

sys.path.append(os.getcwd())

import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import copy
import math
import logging
# import data.experiments
# from utils.utils import re_im, image_json
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

logger = logging.getLogger(__name__)
# helper functions
def rotate(X, center, alpha):
    XX = torch.zeros_like(X)

    XX[:, 0] = (X[:, 0] - center[0]) * np.cos(alpha) + (X[:, 1] - center[1]) * np.sin(alpha) + center[0]
    XX[:, 1] = - (X[:, 0] - center[0]) * np.sin(alpha) + (X[:, 1] - center[1]) * np.cos(alpha) + center[1]

    return XX

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
def flatten(l):
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self,
                 data,
                 save=False,
                 load_p=True,
                 dataset_name="lyft",
                 phase="test",
                 scene_batching = False,
                 obs_len=11,
                 pred_len=50,
                 time_step=0.4,
                 skip=1,
                 data_augmentation=0,
                 scale_img=True,
                 cnn=True,
                 max_num=None,
                 load_semantic_map=False,
                 logger=logger,
                 special_scene=None,
                 scaling_global=1,
                 scaling_local=0.2,
                 grid_size_in_global=12,
                 grid_size_out_global=12,
                 grid_size_local=8,
                 debug=False,
                 img_scaling=0.2,
                 format="units",
                 **kwargs
                 ):
        super().__init__()
        self.__dict__.update(locals())
        self.data = data
        self.train_iters = int(len(self.data)*0.0005)
        self.test_iters = 1000
        if self.phase == "train":
            self.seq_start_end = [(i, i+1) for i in range(self.train_iters)]
        else:
            self.seq_start_end = [(i, i+1) for i in range(self.test_iters)]
#         print(len(self.seq_start_end))

        self.collect_data()
        
    def __len__(self):
        if(self.phase == "train"):
            return self.train_iters
        else:
            return self.test_iters

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        obs_traj = self.obs_traj[start:end]
        pred_traj  = self.pred_traj[start:end]
        obs_traj_rel = self.obs_traj_rel[start:end]
        pred_traj_rel = self.pred_traj_rel[start:end]

        scene = self.scene_list[index]
        current_scene_image =  self.images[scene]

#         if self.wall_available:
#             walls = self.walls_list[index]
#         else:
#             walls = False
        
        walls = self.target_availabilities[start:end]

        if self.data_augmentation:
            obs_traj,pred_traj,obs_traj_rel, pred_traj_rel,  \
            scene_img, global_patch, prob_mask, walls, local_patch = self.data_aug_func(scene = scene,
                                                                                             current_scene_image = current_scene_image,
                                                                                              obs_traj = obs_traj,
                                                                                              pred_traj = pred_traj,
                                                                                              obs_traj_rel = obs_traj_rel,
                                                                                              pred_traj_rel = pred_traj_rel,
                                                                                              walls = walls)
        else:

            scene_img =  [self.image_list[index]]

#             if self.wall_available:

#                 walls = self.walls_list[index]

#             else:
#                 walls = False



            if self.cnn:
                global_patch = self.global_patches[start:end]
                local_patch = self.local_patches[start:end]
                prob_mask = self.Prob_Mask[start:end]
            else:
                global_patch = torch.empty(1)
                local_patch = torch.empty(1)
                prob_mask =  torch.empty(1)
                
        return [obs_traj,
                pred_traj,
                obs_traj_rel,
                pred_traj_rel,
                scene_img,
                global_patch,
                prob_mask,
                walls,
                local_patch
                ]

    def get_stats(self):

#         self.logger.info("Number of trajectories: {}".format(self.num_seq))

        max_dist_obs, _ = torch.abs(self.obs_traj - self.obs_traj[:, -1].unsqueeze(1)).view(-1, 2).max(0)
        max_dist_pred, _ = torch.abs(self.pred_traj - self.obs_traj[:, -1].unsqueeze(1)).view(-1, 2).max(0)
        print('max dist', max_dist_obs)
        print('max dist pred', max_dist_pred)
        

        range_goal = self.grid_size_in_global * self.scaling_global

        self.logger.info(
            "Max Dist Obs: {}, Max Dist Pred: {}, Grid Size: {} ".format(max_dist_obs, max_dist_pred, range_goal))

        if (range_goal < max_dist_obs).any():
            self.logger.warning("Image Patch does not fit all Observations")
        if (range_goal < max_dist_pred).any():
            self.logger.warning("Image Patch does not fit all Predictions")
        max_dx_obs, _ = torch.abs(self.obs_traj_rel).view(-1, 2).max(0)
        max_dx_pred, _ = torch.abs(self.pred_traj_rel).view(-1, 2).max(0)
        print('max_dx_obs', max_dx_obs)
        print('max_dx_pred', max_dx_pred)

        range_goal_local = self.grid_size_local * self.scaling_local

        if (range_goal_local < max_dx_obs).any():
            self.logger.warning("Goal Local Image Patch does not fit all Observations")
        if (range_goal_local < max_dx_pred).any():
            self.logger.warning("Goal Local Image Patch does not fit all Predictions")

        self.logger.info(
            "Max dx Obs: {}, Max dx Pred: {}, Grid Size: {}".format(max_dx_obs, max_dx_pred, range_goal_local))
        self.logger.info(
            "Max dx Obs: {}, Max dx Pred: {}, Grid Size: {}".format(max_dx_obs, max_dx_pred, range_goal_local))

    def gen_global_patches(self, scene_image, trajectory, prediction, image_type="global_image"):
        if self.format == "units":
            scale = 1. / self.scaling_global

        else:
            scale = 1

        rel_scaling = (2 * self.grid_size_in_global + 1) / (2 * self.grid_size_out_global + 1)

        img = scene_image[image_type]

        center_meter = trajectory[-1] 
        end_dist_meters = prediction[scene_image["stop_index"]] - center_meter
        end_point_pixel_global = scale * end_dist_meters

        center_pixel_global = center_meter * scale

        center_scaled = center_pixel_global.long()

        x_center, y_center = center_scaled
        
        cropped_img = TF.crop(img, int(y_center - self.grid_size_in_global), int(x_center - self.grid_size_in_global),
             int(2*  self.grid_size_in_global + 1), int(2* self.grid_size_in_global + 1))

        end_point = end_point_pixel_global / rel_scaling + self.grid_size_out_global

        x_end, y_end = np.clip(int(end_point[0]), 0, 2 * self.grid_size_out_global), np.clip(int(end_point[1]), 0,
                                                                                             2 * self.grid_size_out_global)

        prob_mask = torch.zeros((1, 1, self.grid_size_out_global * 2 + 1, self.grid_size_out_global * 2 + 1)).float()
        prob_mask[0, 0, y_end, x_end] = 1

        position = torch.zeros(1, self.grid_size_in_global * 2 + 1, self.grid_size_in_global * 2 + 1)
        position[0, self.grid_size_in_global, self.grid_size_in_global] = 1

        img = -1 + transforms.ToTensor()(cropped_img)  * 2.

        img = torch.cat((img, position), dim=0).unsqueeze(0)


        return img, prob_mask

    def gen_local_patches(self, scene_image, trajectory, prediction, image_type="local_image"):
        if self.format == "meters":
            scale = 1. / self.scaling_local
        else:
            scale = 1

        img = scene_image[image_type]
        center_meter = trajectory  # center in meter

        end_dist_meters = prediction - center_meter
        end_point_pixel_global = scale * end_dist_meters

        center_pixel_global = center_meter * scale

        center_scaled = center_pixel_global.long()

        x_center, y_center = center_scaled

        cropped_img = TF.crop(img,  int(y_center - self.grid_size_local),int(x_center - self.grid_size_local),
             int(2 *  self.grid_size_local + 1), int(2* self.grid_size_local + 1))

        end_point = end_point_pixel_global + self.grid_size_local

        x_end, y_end = np.clip(int(end_point[0]), 0, 2 * self.grid_size_local), np.clip(int(end_point[1]), 0,
                                                                                        2 * self.grid_size_local)

        prob_mask = torch.zeros((1, self.grid_size_local * 2 + 1, self.grid_size_local * 2 + 1, 1))
        prob_mask[0, y_end, x_end, 0] = 1.

        position = torch.zeros((1, self.grid_size_local * 2 + 1, self.grid_size_local * 2 + 1))
        position[0, self.grid_size_local, self.grid_size_local] = 1.


        img = -1 + transforms.ToTensor()(cropped_img) * 2.
        img = torch.cat((img.float(), position), dim=0).unsqueeze(0)
        
        return img

    def get_local_patches(self):
        feature_list = []
        for index in range(len(self.image_list)):
            scene_image = self.image_list[index]
            trajectories = torch.cat((self.obs_traj[index], self.pred_traj[index]), dim=0)
            feature_pred = []
            for time in range(self.pred_len):
                feature_pred.append(self.gen_local_patches(scene_image, trajectories[self.obs_len - 1 + time],trajectories[time + self.obs_len]))
            feature_pred = torch.cat(feature_pred)
            feature_list.append(feature_pred)
        self.local_patches = torch.stack(feature_list)

    def get_global_patches(self):
        feature_list = []
        cropped_img_list = []
        prob_mask_list = []
        for index in range(len(self.image_list)):
            scene_image = self.image_list[index]
            # for index in np.arange(start, end):
            features, prob_mask = self.gen_global_patches(scene_image, self.obs_traj[index],
                                                                        self.pred_traj[index])
            feature_list.append(features)

            prob_mask_list.append(prob_mask)

        self.global_patches = torch.cat(feature_list)
        self.Prob_Mask = torch.cat(prob_mask_list)


    def get_images_lyft(self, agent):
#         image = agent['image'][24]
        scene = agent['track_id']
        im = agent["image"].transpose(1, 2, 0)
        image = self.data.rasterizer.to_rgb(im)
        img = Image.fromarray(image, 'RGB')

        width = img.size[0]
        height = img.size[1]

        scaled_img = img

        scale_factor = 1
        ratio = 1.

        scale_factor_global = self.img_scaling / self.scaling_global
        global_width = int(round(width * scale_factor_global))
        global_height = int(round(height * scale_factor_global))
        global_image = scaled_img.resize((global_width, global_height), Image.ANTIALIAS)

        scale_factor_local = self.img_scaling / self.scaling_local
        local_width = int(round(width * scale_factor_local))
        local_height = int(round(height * scale_factor_local))
        local_image = scaled_img.resize((local_width, local_height), Image.ANTIALIAS)
        

        self.images.update({scene: {"ratio": ratio, "scale_factor": scale_factor, "scaled_image": scaled_img,
                                    "global_image": global_image, "local_image": local_image}})
    
    def scaleToUnits(self):
        self.obs_traj *= self.img_scaling
        self.pred_traj *= self.img_scaling
        self.obs_traj_rel *= self.img_scaling
        self.pred_traj_rel *= self.img_scaling
#         self.trajectory *= self.img_scaling
        self.format = "units"
        
    def convert_pos(self, agent):
        target_positions = transform_points(agent["target_positions"] + agent["centroid"][:2], agent["world_to_image"])
        hist_positions = transform_points(agent["history_positions"] + agent["centroid"][:2], agent["world_to_image"])
        try:
            stop_index = np.where(agent['target_availabilities'] == 0)[0][0]
        except:
            stop_index = len(agent['target_availabilities'])-1
        return hist_positions, target_positions, stop_index

    def collect_data(self):
        self.wall_available = False
        self.obs_traj = []
        self.pred_traj = []
        self.obs_traj_rel = []
        self.pred_traj_rel = []
        self.image_list = []
        self.images = {}
        self.scene_list = []
        
        self.wall_points_dict = {}
        self.walls_list = []
        self.target_availabilities = []
        if self.phase == "train":
            ind = np.array([i for i in range(self.train_iters)])
        else:
            ind = np.array([i for i in range(self.train_iters,self.train_iters+self.test_iters)])
                
        for i in tqdm_notebook(ind): 
            agent = self.data[i]
            scene = agent['track_id']
            self.scene_list.append(scene)
            past, future, stop_index_target = self.convert_pos(agent)
            past = past[::-1]
            self.obs_traj.append(past)
            self.pred_traj.append(future)
            self.get_images_lyft(agent)
            seq_list_rel_o = past[1:] - past[:-1]
            seq_list_rel_p = future[1:] - future[:-1]
            last_past_first_future_dx = future[0] - past[-1]
            seq_list_rel_p = np.vstack((last_past_first_future_dx, seq_list_rel_p))
            self.image_list.append({"ratio": self.images[scene]["ratio"], "scene": scene, "scaled_image": self.images[scene]["scaled_image"],
                                     "global_image": self.images[scene]["global_image"], "local_image": self.images[scene]["local_image"], "stop_index": stop_index_target})
            self.obs_traj_rel.append(seq_list_rel_o)
            self.pred_traj_rel.append(seq_list_rel_p)
            self.target_availabilities.append([stop_index_target])
            
        self.obs_traj = torch.from_numpy(np.array(self.obs_traj)).type(torch.float)
        self.pred_traj = torch.from_numpy(np.array(self.pred_traj)).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(np.array(self.obs_traj_rel)).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(np.array(self.pred_traj_rel)).type(torch.float)
        self.target_availabilities = torch.from_numpy(np.array(self.target_availabilities)).type(torch.int)
        
        
        self.scaleToUnits()

        self.get_global_patches()
        self.get_local_patches()
        if self.save:
            self.save_dset()
#         self.get_stats()

        # return obs_traj_list, pred_traj_list, obs_traj_rel, pred_traj_rel, self.image_list, self.global_patches, self.Prob_Mask, [], self.local_patches
    


def seq_collate(data):
    obs_traj_list, pred_traj_list, obs_traj_rel_list, pred_traj_rel_list, scene_img_list, features_list, prob_mask_list, wall_list, features_tiny_list = zip(*data)
    _len = [len(seq) for seq in obs_traj_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()

    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_traj_list, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_traj_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_traj_rel_list, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_traj_rel_list, dim=0).permute(1, 0, 2)
    wall_list = torch.cat(wall_list, dim=0)

    scene_img_list = tuple(flatten(list(scene_img_list)))
    try:
        global_patch = torch.cat(features_list, dim=0)
        local_patch = torch.cat(features_tiny_list, dim=0)
        prob_mask = torch.cat(prob_mask_list, dim=0)
    except:
        global_patch = torch.empty(1)
        local_patch = torch.empty(1)
        prob_mask = torch.empty(1)

    return {"in_xy": obs_traj,
            "gt_xy": pred_traj,
            "in_dxdy": obs_traj_rel,
            "gt_dxdy": pred_traj_rel,
            "size": torch.LongTensor([obs_traj.size(1)]),
            "scene_img": scene_img_list,
            "global_patch": global_patch,
            "prob_mask": prob_mask,
            "occupancy": wall_list,
            "local_patch": local_patch,
            "seq_start_end": seq_start_end
            }



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session