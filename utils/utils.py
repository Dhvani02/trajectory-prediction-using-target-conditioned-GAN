import pytorch_lightning as pl
import numpy as np
import torch
import random
from matplotlib.path import Path
import numpy as np
import torch.nn as nn
import math
import torch
from torch.optim.optimizer import Optimizer, required
import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import torch.nn as nn
import itertools
import importlib
from attrdict import AttrDict
from argparse import Namespace
import os.path
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json
import random
import os


class BatchSizeScheduler(pl.callbacks.base.Callback):
    """
    Implmentation of a BatchSize Scheduler following the paper
    'Don't Decay the Learning Rate, Increase the Batch Size'
    (https://arxiv.org/abs/1711.00489)
    The scheduler increases the batchsize if the validation loss does not decrease.
    """
    def __init__(self, bs = 4 , factor=2, patience=3, max_bs=64, mode = "min", monitor_val = "val_loss"):
        """
        :param bs: initial batch size
        :param factor: factor by which current batch size is increased
        :param patience: duration in which loss does not have to decrease
        :param max_bs: maximum batch size
        :param mode: considering 'min' or 'max' for 'monitor_val'
        :param monitor_val: considered loss for scheduler
        """

        self.factor = factor
        self.patience = patience
        self.max_bs = max_bs
        self.current_count = patience*1.
        self.cur_metric = False
        self.monitor_metric = monitor_val
        self.cur_bs = bs
        if mode not in ["min", "max"]:
            assert False, "Variable for mode '{}' not valid".format(mode)
        self.mode = mode
        if max_bs > bs:
            self.active = True
        else: self.active = False

    def on_validation_end(self, trainer, pl_module):

        self.cur_bs = int(np.minimum(self.cur_bs * self.factor, self.max_bs))

        # set new batch_size
        pl_module.batch_size = self.cur_bs
        trainer.reset_train_dataloader(pl_module)

        if not self.cur_metric:
            self.cur_metric = trainer.callback_metrics[self.monitor_metric]

        if self.active:
            if self.mode == "min":
                if trainer.callback_metrics[self.monitor_metric]  < self.cur_metric:
                    self.cur_metric =trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience*1

                else:
                    self.current_count-=1


            else:
                if trainer.callback_metrics[self.monitor_metric]  > self.cur_metric:
                    self.cur_metric = trainer.callback_metrics[self.monitor_metric]
                    self.current_count = self.patience*1

                else:
                    self.current_count -= 1

            if self.current_count == 0:
                self.cur_bs = int(np.minimum(self.cur_bs*self.factor, self.max_bs))

                # set new batch_size
                pl_module.batch_size = self.cur_bs
                trainer.reset_train_dataloader(pl_module)
                print("SET BS TO {}".format(self.cur_bs))
                self.current_count = self.patience*1
                if self.cur_bs >=self.max_bs:
                    self.active = False



class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, random_label = 1, delta_rand = 0.15,reduction = "mean" ):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgan.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.register_buffer('delta_rand', torch.tensor(delta_rand))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(reduction = reduction)
            self.register_buffer('random_label', torch.tensor(0))
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction = reduction)
            self.register_buffer('random_label', torch.tensor(random_label))
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.register_buffer('random_label', torch.tensor(0))
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if self.random_label:
            if target_is_real:
                labels = torch.FloatTensor(prediction.size()).uniform_(self.real_label - self.delta_rand, self.real_label)
            else:
                labels = torch.FloatTensor(prediction.size()).uniform_(self.fake_label,self.fake_label +  self.delta_rand)
        else:

            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            labels = target_tensor.expand_as(prediction)
        return labels

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.

        Example:

            criterionGAN = GANLoss(gan_mode).to(device)
            loss_D_real = self.criterionGAN(self.discriminator_realfake(faces), True)  # give True (1) for real samples
            loss_D_fake = self.criterionGAN(self.discriminator_realfake(generated_images), False)  # give False (0) for generated samples
            loss_D = loss_D_real + loss_D_fake

            loss_G = self.criterionGAN(self.discriminator_realfake(generated_images), True)  # give True (1) labels for generated samples, aka try to fool D
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).to(prediction)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction
            else:
                loss = prediction
        return loss

def cal_gradient_penalty(netD, real_data, fake_data , device, type, constant, lambda_gp):

    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
            *real_data.shape)
        alpha = alpha.to(device)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
    return gradient_penalty





def l2_loss(pred_traj, pred_traj_gt, mode='average', type = "mse"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    # disabled loss mask
    #loss_mask = 1
    seq_len, batch, _ = pred_traj.size()
    #loss_mask = loss_mask.permute(2, 1,0)[:, :, 0]
    d_Traj = pred_traj_gt - pred_traj

    if type =="mse": 
        loss = torch.norm( (d_Traj), 2, -1)
    elif type =="average":
      
        loss = (( torch.norm( d_Traj, 2, -1)) + (torch.norm( d_Traj[-1], 2, -1)))/2.
       
    else: 
        raise AssertionError('Mode {} must be either mse or  average.'.format(type))


    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
     
        return torch.mean(loss, dim =0 )
    elif mode == 'raw':
        return loss.sum(dim=0)


def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - mode: Can be one of sum, average, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, num_traj, _ = pred_traj.size()
    
    loss = pred_traj_gt - pred_traj

    loss = torch.norm(loss, 2, 2).unsqueeze(0)
    
    
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average': 
        return torch.sum(loss)/( seq_len * num_traj)
    elif mode == 'raw':
     
        return torch.sum(loss, 1)


def final_displacement_error(
    pred_pos, pred_pos_gt, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos

    Output:
    - loss: gives the eculidian displacement error
    """
    num_traj, _ = pred_pos_gt.size() 
    loss = pred_pos_gt - pred_pos

    loss = torch.norm(loss, 2, 1).unsqueeze(0)
    if mode == 'raw':
        
        return loss
    elif mode == 'average':
        return torch.sum( loss) / num_traj
    elif mode == 'sum':
        return torch.sum(loss)


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, mode='sum'
    )
   
    return g_l2_loss_abs

    


def cal_ade(pred_traj_gt, pred_traj_fake, mode = "sum"):

    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode = mode)

    return ade


def cal_fde(
    pred_traj_gt, pred_traj_fake, mode = "sum"):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1],  mode = mode )
    return fde


def crashIntoWall( traj, walls):
    length, batch, dim = traj.size()


    wall_crashes = []
    for i in range(batch):
        t = traj[:, i, :]
        for wall in walls[i]:

            polygon = Path(wall)

            wall_crashes.append(1* polygon.contains_points(t).any())
    return wall_crashes

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_( grad, grad, value =  1 - beta2)
                exp_avg.mul_(beta1).add_( grad,  alpha = 1 - beta1)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_( p_data_fp32, alpha = -group['weight_decay'] * group['lr'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_( exp_avg, denom, value = -step_size * group['lr'])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_( p_data_fp32, alpha = -group['weight_decay'] * group['lr'])
                    p_data_fp32.add_(exp_avg, alpha = -step_size * group['lr'])
                    p.data.copy_(p_data_fp32)

        return loss


class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def distanceP2W(point, wall):
    p0 = np.array([wall[0], wall[1]])
    p1 = np.array([wall[2], wall[3]])

    d = p1 - p0
    ymp0 = point - p0
    t = np.dot(d, ymp0) / np.dot(d, d)
    if t > 0.0 and t < 1.:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point)

    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point) * 0

    return dist, npw


def image_json(scene, json_path, scaling=1):
    json_path = os.path.join(json_path, "{}_seg.json".format(scene))

    wall_labels = ["lawn", "building", "car", "roundabout"]

    walls = []
    wall_points = []

    start_end_points = {}
    decisionZone = {}
    directionZone = {}


    nr_start_end = 0

    with open(json_path) as json_file:

        data = json.load(json_file)
        for p in data["shapes"]:
            label = p["label"]

            if label in wall_labels:

                points = np.array(p["points"]).astype(int)

                points = order_clockwise(points)
                for i in np.arange(len(points)):
                    j = (i + 1) % len(points)

                    p1 = points[i]
                    p2 = points[j]

                    concat = np.concatenate((p1, p2))
                    walls.append(scaling * concat)

                wall_points.append([p * scaling for p in points])
            elif "StartEndZone" in label:
                id = int(label.split("_")[-1])
                start_end_points[nr_start_end] = {"point": scaling * np.array(p["points"]),
                                                  "id": id}
                nr_start_end += 1
            elif "decisionZone" in label:
                id = int(label.split("_")[-1])
                decisionZone[id] = scaling * np.array(p["points"])

            elif "directionZone" in label:
                id = int(label.split("_")[-1])
                directionZone[id] = scaling * np.array(p["points"])

    return walls, wall_points, start_end_points, decisionZone, directionZone


# order points clockwise

def order_clockwise(point_array, orientation=np.array([1, 0])):
    center = np.mean(point_array, axis=0)
    directions = point_array - center

    angles = []
    for d in directions:
        t = np.arctan2(d[1], d[0])
        angles.append(t)
    point_array = [x for _, x in sorted(zip(angles, point_array))]

    return point_array


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            break

    return random_point


def get_batch_k( batch, k):
    new_batch = {}
    for name, data in batch.items():

        if name in ["global_patch", "prob_mask"]:

            new_batch[name] = data.repeat(k, 1, 1, 1).clone()
        elif name in ["local_patch"]:

            new_batch[name] = data.repeat(k, 1, 1, 1, 1).clone()
        elif name in ["scene_img"]:

            new_batch[name] = data * k
        elif name not in ["size", "scene_nr", "scene", "img", "cropped_img", "seq_start_end"]:
            new_batch[name] = data.repeat(1, k, 1).clone()


        else:
            new_batch[name] = data
    return new_batch



def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed



def re_im(img):
    """ Rescale images """
    img = (img + 1)/2.
    return img

import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import glob
import os
import random
import matplotlib.animation as animation
""" This code contains visualization functions to generate GIFs. Code is messy, use code at your own risk"""


def make_gif(y_softmax=None,
			 y=None,
			 global_patch=None,
			 grid_size_in_global=None,
			 probability_mask=None,
			 return_mode="buf",
			 axs=None,
			 input_trajectory=None,
			 gt_trajectory=None,
			 prediction_trajectories=None,
			 background_image=None,
			 img_scaling=None,
			 final_position = None,
			 scaling_global=None,
			 num_traj = 1

			 ):

	def set_axis(ax, legend_elements):
		plt.axis('off')
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)

		ax.set_xlim(extent[0], extent[1])
		ax.set_ylim(extent[3], extent[2])
		legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol = 3, framealpha=0, columnspacing =1, handletextpad= 0.4)

		plt.setp(legend.get_texts(), color='w')
		buf = io.BytesIO()

		plt.savefig(buf, bbox_inches='tight', format='jpeg', dpi=300)
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)

		return image
	from PIL import Image, ImageDraw
	from matplotlib.lines import Line2D



	if img_scaling:
		if not type(input_trajectory) == type(None): input_trajectory /= img_scaling
		if not type(prediction_trajectories) == type(None): prediction_trajectories /= img_scaling
		if not type(gt_trajectory) == type(None): gt_trajectory /= img_scaling
		if not type(final_position) == type(None): final_position /= img_scaling
	images = []

	color = "red"
	color_goal = "red"
	marker_input = "o"
	marker_output = "v"
	marker_goal = "*"
	marker_size = 3



	legend_elements = [Line2D([0], [0], marker = marker_input ,color='w', lw=0 ,  markerfacecolor = color ,markeredgecolor=color,  label="Input") ,
					   Line2D([0], [0], marker=marker_output,color='w', lw=0 , markerfacecolor=color, markeredgecolor=color, label="Prediction"),
					   Line2D([0], [0], marker=marker_goal,color='w', lw=0 , markerfacecolor=color_goal, markeredgecolor=color_goal, label="Goal")]



	center = input_trajectory[-1, 0]
	dx = (grid_size_in_global + 0.5) / img_scaling


	extent = [center[0] - dx, center[0] + dx \
		, center[1] + dx, center[1] - dx]

	if any(np.array(extent) < 0):
		return 0

	final_position += center.view(1, 1, -1)


	max_dist = 0
	for j in range(1, final_position.size(1)):
		dist = torch.norm( final_position[0, 0]-final_position[0, j], 2)

		if dist > max_dist:
			max_dist = dist
			index = j

	trajectories_id = [0, index]

	for j in range(1, final_position.size(1)):
		if j != index:
			trajectories_id.append(j)
		if len(trajectories_id) == num_traj:
			break


	for num  in trajectories_id:

		for i in range(2, len(input_trajectory)):
			fig, ax = plt.subplots()

			plt.imshow(background_image)
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size, label="Input")

			images.append(set_axis(ax, legend_elements))

		softmax_img = np.clip(y_softmax[num][0 ].cpu().numpy(), 0, 1)
		softmax_img /= np.max(softmax_img)
		softmax_img = Image.fromarray(softmax_img)
		softmax_img = ToTensor()(np.array(
			softmax_img.resize(
				(int(2*dx), int(2 * dx)), resample = PIL.Image.LANCZOS)))


		color_map = (torch.tensor([1., 1., 0.]).view(-1, 1, 1) - torch.ones(3, int(2*dx), int(2 * dx)) * softmax_img  * torch.tensor([0., 1., 0.]).view(-1, 1, 1) )
		softmax_img = torch.cat((color_map, softmax_img), dim=0).permute(1, 2, 0)
		final_softmax_image = torch.zeros(background_image.width, background_image.height, 4)

		final_softmax_image[int(extent[3]) :int(extent[3]) +int(2*dx), int(extent[0]):int(extent[0])+ int(2*dx) ] = softmax_img
		time_duration = 5


		# print(softmax_img.size() )


		## Show probability
		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image)

			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size)



			images.append(set_axis(ax, legend_elements))

		## Show probability
		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image, interpolation = "lanczos")

			plt.plot(final_position[0, num , 0], final_position[0, num , 1], marker=marker_goal, color=color_goal, label="Goal ")
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num , 1], marker=marker_input, color=color,
						 markersize=marker_size)

			images.append(set_axis(ax, legend_elements))

		for t in range(time_duration):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			ax.imshow(final_softmax_image, interpolation = "lanczos")


			plt.plot(final_position[0, num , 0], final_position[0, num, 1], marker=marker_goal, color=color_goal)
			for k in range(3):
				plt.plot(input_trajectory[i - k, num , 0], input_trajectory[i - k, num, 1], marker=marker_input, color=color,
						 markersize=marker_size)

			images.append(set_axis(ax, legend_elements))
		# PREDICTION
		for i in range(0, len(prediction_trajectories)):
			fig, ax = plt.subplots()
			ax.imshow(background_image)
			plt.plot(final_position[0, num , 0], final_position[0, num , 1], marker=marker_goal, color=color_goal)
			for k in range(3):

				ind = i - k
				if ind < 0:
					plt.plot(input_trajectory[ind, num , 0], input_trajectory[ind, num , 1], marker=marker_output, color=color,
							 markersize=marker_size)
				else:
					plt.plot(prediction_trajectories[ind, num , 0], prediction_trajectories[ind, num , 1], marker=marker_output,
							 color=color, markersize=marker_size)

			images.append(set_axis(ax, legend_elements))

	GIF_DIRECTORY = "images/gif"
	gifs = glob.glob(os.path.join(GIF_DIRECTORY, "*.gif"))
	import re
	if len(gifs) < 1:
		number = [-1]
	else:
		number = [int(re.search(r'\d+', file).group()) for file in  gifs]
	# from IPython.display import display
	# display(images[0])
	# display(images[10])
	# print(images)
	# images[0].save('images/gif/gif_{}'.format(max(number)+1), "JPEG", quality=10  )
	#
	# fdfh
	#

	images[0].save('images/gif/gif_{}.gif'.format(max(number)+1),
				   save_all=True, append_images=images[1:], duration=200, loop=0)

	print("saved")
def visualize_probabilities(y_softmax = None,
							y = None,
							global_patch = None,
							grid_size_in_global = None,
							probability_mask = None,
							return_mode = "buf",
							axs = None):

	""""""
	color = torch.ones(3, int(2 * grid_size_in_global + 1),
					   int(2 * grid_size_in_global + 1)) * torch.tensor([1., 0., 0.]).view(-1, 1, 1)
	recon_img = np.clip(y[0][0].cpu().numpy(), 0, 1)
	recon_img = Image.fromarray(recon_img)
	recon_img = ToTensor()(np.array(recon_img.resize(
		(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))
	recon_img = torch.cat((color, recon_img.cpu()), dim=0).permute(1, 2, 0)

	softmax_img = np.clip(y_softmax[0][0].cpu().numpy(), 0, 1)
	softmax_img /= np.max(softmax_img)
	softmax_img = Image.fromarray(softmax_img)
	softmax_img = ToTensor()(np.array(
		softmax_img.resize(
			(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))

	softmax_img = torch.cat((color, softmax_img.cpu()), dim=0).permute(1, 2, 0)

	gt_img = np.clip(probability_mask, 0, 1)
	gt_img = Image.fromarray(gt_img)
	gt_img = ToTensor()(np.array(gt_img.resize(
		(int(2 * grid_size_in_global + 1), int(2 * grid_size_in_global + 1)))))
	gt_img = torch.cat((color, gt_img.cpu()), dim=0).permute(1, 2, 0)

	real_img = np.clip(global_patch, 0, 1)

	if axs is None:
		fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

	axs[0].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[0].imshow(softmax_img, interpolation='nearest')
	axs[0].set_title("Probability Map")
	axs[0].axis('off')

	axs[1].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[1].imshow(recon_img, interpolation='nearest', )
	axs[1].set_title("Goal Realisation")
	axs[1].axis('off')

	axs[2].imshow(np.transpose(real_img, (1, 2, 0)), interpolation='nearest')#,  origin='upper')
	axs[2].imshow(gt_img, interpolation='nearest')
	axs[2].set_title("GT Goal")
	axs[2].axis('off')

	if return_mode == "buf":
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	elif return_mode == "ax":
		return axs
	else:
		return fig

def visualize_trajectories(input_trajectory = None,
						   gt_trajectory = None,
						   prediction_trajectories = None,
						   background_image = None,
						   img_scaling = None,
						   scaling_global = None,
						   grid_size = 20 ,
						   return_mode = "buf",
						   axs = None):



	# re-scale trajectories
	if img_scaling:
		if not type(input_trajectory) == type(None): input_trajectory/=img_scaling
		if not type(prediction_trajectories) == type(None): prediction_trajectories/=img_scaling
		if not type(gt_trajectory) == type(None): gt_trajectory/= img_scaling

	center = input_trajectory[-1]
	dx = (grid_size + 0.5)/img_scaling*scaling_global
	if axs is None:
		fig_trajectory, axs = plt.subplots()

	axs.imshow(background_image,  alpha=0.9)

	extent = [center[0] - dx, center[0] + dx \
		, center[1] + dx, center[1] - dx]

	axs.scatter(input_trajectory[:, 0], input_trajectory[:, 1], color="black", marker="o", s=2)
	marker_style = dict(color='tab:orange', marker='o',
						markersize=3, markerfacecoloralt='tab:red', markeredgewidth=1)

	for i in range(prediction_trajectories.size(1)):
	 	#ax_trajectory.plot(prediction_trajectories[:, i, 0], prediction_trajectories[:, i,1], linestyle='None', fillstyle="none", **marker_style)
		axs.plot(prediction_trajectories[:, i, 0], prediction_trajectories[:, i, 1], color="red")

	axs.axis('off')
	#
	# axs.set_xlim(extent[0] - grid_size, extent[1] + grid_size)
	# axs.set_ylim(extent[3] - grid_size, extent[2] + grid_size)

	if return_mode == "buf":
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	elif return_mode=="ax":
		return axs
	else:
		return fig_trajectory


def visualize_traj_probabilities(input_trajectory=None,
							   gt_trajectory=None,
							   prediction_trajectories=None,
							   background_image=None,
							   img_scaling=None,
							   scaling_global=None,
							   grid_size=20,
								y_softmax = None,
								 y=None,
								 global_patch=None,
								 grid_size_in_global=None,
								 probability_mask=None,
								 buf="buf"):
	fig, axs = plt.subplots(1, 4, figsize=(16, 4))


	visualize_probabilities(y_softmax = y_softmax,
							y = y,
							global_patch = global_patch,
							grid_size_in_global = grid_size_in_global,
							probability_mask = probability_mask,
							return_mode = "ax",
							axs = axs[1:])

	visualize_trajectories(input_trajectory = input_trajectory,
							   gt_trajectory = gt_trajectory,
							   prediction_trajectories = prediction_trajectories,
							   background_image = background_image,
							   img_scaling = img_scaling,
							   scaling_global = scaling_global,
							   grid_size = 20,
								return_mode = "ax",
								axs=axs[0])




	if buf:
		buf = io.BytesIO()
		plt.savefig(buf, format='jpeg')
		plt.close()
		buf.seek(0)
		image = PIL.Image.open(buf)
		image = ToTensor()(image)
		return image
	else:
		return fig_trajectory





def make_gif_dataset(dataset, k =20):

	def set_axis(ax, legend_elements):
		plt.axis('off')
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3,
						   framealpha=0, columnspacing=1, handletextpad=0.4)

		plt.setp(legend.get_texts(), color='w')
		# buf = io.BytesIO()
		#
		# plt.savefig(buf, bbox_inches='tight', format='jpeg')
		# plt.close()
		# buf.seek(0)
		# image = Image.open(buf).convert('P')

		return ax
	from PIL import Image, ImageDraw
	from matplotlib.lines import Line2D

	# Tableau 20 Colors
	tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
				 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
				 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
				 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
				 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]



	for i in range(len(tableau20)):
		r, g, b = tableau20[i]
		tableau20[i] = (r / 255., g / 255., b / 255.)
	random.shuffle(tableau20)
	img_scaling = dataset.img_scaling


	marker_input = "o"
	marker_output = "v"


	range_list = list(range(200))
	random.shuffle(range_list)
	fig = plt.figure()
	img_id = 0
	for c_id, index in enumerate(range_list[:k]):
		item = dataset.__getitem__(index)
		print(index)
		input_trajectory = item[0]
		gt_trajectory = item[1]
		if img_scaling:
			if not type(input_trajectory) == type(None): input_trajectory /= img_scaling
			if not type(gt_trajectory) == type(None): gt_trajectory /= img_scaling

		legend_elements = [
			Line2D([0], [0], marker=marker_input, color='w', lw=0, markerfacecolor=tableau20[c_id], markeredgecolor=tableau20[c_id],
				   label="Input"),
			Line2D([0], [0], marker=marker_output, color='w', lw=0, markerfacecolor=tableau20[c_id], markeredgecolor=tableau20[c_id],
				   label="Ground-truth")]

		for traj in input_trajectory[0]:
			f , ax = plt.subplots()
			ax.imshow(item[4][0]["scaled_image"])
			im = ax.plot(traj[0], traj[1], marker=marker_input, color = tableau20[c_id])
			set_axis(ax, legend_elements)
			# plt.show()
			plt.tight_layout()
			plt.savefig("images/gif_dataset/{}.jpg".format(img_id), bbox_inches='tight', pad_inches=0,  dpi = 300)
			
			img_id += 1
			plt.close()

		for traj in gt_trajectory[0]:
			f , ax = plt.subplots()
			ax.imshow(item[4][0]["scaled_image"])
			plt.plot(traj[0], traj[1], marker=marker_output, color = tableau20[c_id])
			set_axis(ax, legend_elements)
			# plt.show()
			plt.tight_layout()
			plt.savefig("images/gif_dataset/{}.jpg".format(img_id), bbox_inches='tight', pad_inches=0, dpi=300)

			img_id+=1
			plt.close()
	