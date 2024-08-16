import datetime
import sys
sys.path.append('../PanoHead')
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import dnnlib
import random
from PIL import Image

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from torchvision import models
from torchvision.transforms import InterpolationMode
from dataset import FSPairedDataset, BezierPointsDataset, ResterDataset
from model.Encoder import Encoder

# loss
from criteria import id_loss, w_norm, moco_loss
from criteria.lpips.lpips import LPIPS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='lr')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--resolution', type=int, default=512, help='image resolution')
parser.add_argument('--target', type=str, default='ws', help='target feature')
parser.add_argument('--save_minibatch_freq', type=int, default=1000, help='set minibatch frequency to save model')
parser.add_argument('--show_minibatch_freq', type=int, default=10, help='set minibatch frequency to calculate loss')
parser.add_argument('--checkpoints_root', type=str, default='./checkpoints', help='checkpoints root')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--experiment', type=str, default='change experiment name', help='experiment name')
parser.add_argument('--encoder_path', type=str, default=None, help='encoder path to load')
parser.add_argument('--cos_sim_flag', type=bool, default=False, help='cos_sim_flag')
parser.add_argument('--is_bezier', type=bool, default=False, help='is_bezier')

# loss
parser.add_argument('--loss_base_lambda', type=int, default=1.0, help='loss_base_lambda')

parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
# parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
parser.add_argument('--lpips_lambda_crop', default=0, type=float, help='LPIPS loss multiplier factor for inner image region')
parser.add_argument('--l2_lambda_crop', default=0, type=float, help='L2 loss multiplier factor for inner image region')
parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')

# panohead
parser.add_argument('--panohead_path', type=str, default='../PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl', help='path of pretrained PanoHead Generator')
parser.add_argument('--reload_modules', type=bool, default = True, help='Overload persistent modules?')
parser.add_argument('--fov_deg', type=int, default = 18.837, help='Field of View of camera in degrees')
parser.add_argument('--pose_cond', type=int, default = 90, help='camera conditioned pose angle')
parser.add_argument('--trunc', type=float, default = 0.7, help='Truncation psi')
parser.add_argument('--trunc_cutoff', type=int, default = 14, help='Truncation cutoff')

# params
opt = parser.parse_args()
print(opt)

epoch = opt.epoch
device = opt.device
lr = opt.lr
size = opt.resolution
target = opt.target
save_minibatch_freq = opt.save_minibatch_freq
show_minibatch_freq = opt.show_minibatch_freq
batch_size = opt.batch
experiment = opt.experiment
encoder_path = opt.encoder_path
cos_sim_flag = opt.cos_sim_flag
is_bezier = opt.is_bezier

# loss param
loss_base_lambda = opt.loss_base_lambda

lpips_lambda = opt.lpips_lambda
id_lambda = opt.id_lambda
l2_lambda = opt.l2_lambda
# w_norm_lambda = opt.w_norm_lambda
lpips_lambda_crop = opt.lpips_lambda_crop
l2_lambda_crop = opt.l2_lambda_crop
moco_lambda = opt.moco_lambda

# panohead
panohead_path = opt.panohead_path
reload_modules = opt.reload_modules
fov_deg = opt.fov_deg
pose_cond = opt.pose_cond
truncation_psi = opt.trunc
truncation_cutoff = opt.trunc_cutoff

scaler = GradScaler()
# tensorboard
# cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'logs/{experiment}/'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# checkpoint
checkpoints_root = opt.checkpoints_root
checkpoint_dir = os.path.join(checkpoints_root, experiment)
os.makedirs(checkpoint_dir, exist_ok=True)

# model 
if encoder_path == None:
    net = Encoder(target, is_bezier).to(device)
    print("init encoder!!")
else:
    net = Encoder(target, is_bezier).to(device)
    net.load_state_dict(torch.load(encoder_path))
    print("load encoder successfully!!")
print('Loading panohead networks from "%s"...' % panohead_path)
with dnnlib.util.open_url(panohead_path) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
# Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
if reload_modules:
    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

# optim & scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
lr_scheduler= torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

# Initialize loss
if id_lambda > 0 and moco_lambda > 0:
    raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

mse_loss = nn.MSELoss().to(device).eval()
if lpips_lambda > 0:
    lpips_loss = LPIPS(net_type='alex').to(device).eval()
if id_lambda > 0:
    id_loss = id_loss.IDLoss().to(device).eval()
# if w_norm_lambda > 0:
#     w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=start_from_latent_avg)
if moco_lambda > 0:
    moco_loss = moco_loss.MocoLoss().to(device).eval()


# dataset
transform = transforms.Compose([
                # transforms.Resize(int(size * 1.12), InterpolationMode.BICUBIC),
                # transforms.RandomCrop(size),
                transforms.ToTensor()])
if is_bezier:
    dataset = BezierPointsDataset('./data', None, target = target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    dataset = ResterDataset('./data', None, transform = transform, target = target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_batch = len(dataloader)
###### Training ######
running_loss = 0.0
pose_cond_rad = pose_cond/180*np.pi
intrinsics = FOV_to_intrinsics(fov_deg, device=device)

for cur_epoch in range(epoch):
    for i, batch in enumerate(dataloader):  # len(dataset): 100,000
        
        if is_bezier:
            data  = batch['b_point'].cuda()
        else: 
            data  = batch['img'].cuda()
        minibatch_size = data.shape[0]
        camera_angle = [(ang_y, ang_p) for ang_y, ang_p in zip(batch['angle'][0], batch['angle'][1])]
        # [(ang1_y, ang2_y, ang3_y, ang4_y, ....batch size), (ang1_p, ang2_p, ang3_p, ang4_p, .... batch size)]
        camera_angle_idx = batch['angle_idx'].cuda()

        seed = batch['seed']
        loss_recon_batch = 0
        # data: [batch_size, 3, 256, 256]    

        # set seed fix
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)        

        if target == 'z':
            target_f = batch['target'].float().cuda()
        else:
            target_f  = batch['target'].cuda()
        # z: [batch_size, 1, 512]
        # ws: [batch_size, 1, 14, 512]
        # tri_plane: [batch_size, 3, 96, 256, 256]

        #################### Encoder ####################
        optimizer.zero_grad()
        with autocast():
            net.train()
            feature = net(data)

        #################### Regenerate ####################
        # load PanoHead 
        cam_pivot = torch.tensor([0, 0, 0], device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        
        # set ws depending on target feature 
        if target == 'ws':
            ws = feature.unsqueeze(dim=1)    # [4, 1, 14, 512]
            ws_chunk =  torch.split(ws, 1, dim=0) # 4 * [1, 1, 14, 512]
        elif target == 'tri_plane':
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
            z = torch.split(z, 1, dim=0)
            ws_chunk = [G.mapping(single_z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff).unsqueeze(0) 
                  for single_z in z]
            # split feature
            tri_plane_chunk = torch.split(feature, 1, dim=0)
        elif target == 'z':
            z = torch.split(feature.squeeze(dim=1), 1, dim=0)
            ws_chunk = [G.mapping(single_z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff).unsqueeze(0) 
                  for single_z in z]
                        
        for batch_idx in range(minibatch_size):
            angle_y = camera_angle[batch_idx][0]
            angle_p = camera_angle[batch_idx][1]

            # rand camera setting
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # ws_single = ws_chunk[batch_idx].squeeze(0)

            # tri-plane load
            if target == 'tri_plane':
                G.set_tri_plane(tri_plane_chunk[batch_idx])
            recon_img = G.synthesis(ws_chunk[batch_idx].squeeze(0), camera_params)['image']
            # recon_img = (recon_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            
            recon_img = (recon_img - recon_img.min())/(recon_img.max()-recon_img.min())
            
            # real image gt load 
            real_img_path = f'./data/real_pic/{seed[batch_idx]:05d}/{seed[batch_idx]:05d}-{camera_angle_idx[batch_idx]}.png'
            real_img = transform(Image.open(real_img_path).convert('RGB')).cuda()
            real_img = real_img.unsqueeze(0)
            # loss_recon = 1e-10    
            # with autocast():
            #     # loss_recon
            #     if id_lambda > 0:    # 0
            #         loss_id, _, _ = id_loss(recon_img, real_img)
            #         loss_recon = loss_id * id_lambda
            #     if l2_lambda > 0:   # 1.0
            #         loss_l2 = mse_loss(recon_img, real_img)
            #         loss_recon += loss_l2 * l2_lambda
            #     if lpips_lambda > 0:    # 0.8
            #         loss_lpips = lpips_loss(recon_img, real_img)
            #         loss_recon += loss_lpips * lpips_lambda
            #     if lpips_lambda_crop > 0:   # 0
            #         loss_lpips_crop = lpips_loss(recon_img[:, :, 35:223, 32:220], real_img[:, :, 35:223, 32:220])   # Reconsider crop size of resolution 512 x 512 
            #         loss_recon += loss_lpips_crop * lpips_lambda_crop
            #     if l2_lambda_crop > 0:  # 0
            #         loss_l2_crop = mse_loss(recon_img[:, :, 35:223, 32:220], real_img[:, :, 35:223, 32:220])
            #         loss_recon += loss_l2_crop * l2_lambda_crop
            #     if moco_lambda > 0: # 0
            #         loss_moco, _, id_logs = moco_loss(recon_img, real_img)
            #         loss_recon += loss_moco * moco_lambda
            #     loss_recon_batch += loss_recon
           
            # check recon image
            # recon_img_test = (recon_img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # real_img_test = (real_img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # Image.fromarray(recon_img.cpu().numpy(), 'RGB').save('./sample_recon.png')
            # Image.fromarray(real_img.cpu().numpy(), 'RGB').save('./sample_real.png')
            # print(f"loss_recon:{loss_recon}, loss_l2:{loss_l2}, loss_lpips:{loss_lpips}")
        # loss_recon_batch /= minibatch_size 

        ########## loss calculation and optimization ##########
        with autocast():
            loss_base = mse_loss(feature, target_f.squeeze()) 
        print(f"loss_recon_batch: ,loss_base:{loss_base} ")
        
        loss = loss_base_lambda * loss_base # + loss_recon_batch
        scaler.scale(loss).backward()
        # loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()
        running_loss += loss.item()
        ########## tensorboard log save ##########
        if i % show_minibatch_freq == show_minibatch_freq-1:
            # scalar
            avg_loss = running_loss / show_minibatch_freq
            print(f'Epoch [{cur_epoch+1}/{epoch}], Step [{i+1}/{num_batch}], Loss: {avg_loss:.4f}')
            writer.add_scalar('training loss', avg_loss, cur_epoch * num_batch + i)
            running_loss = 0.0
            
            # image(real image, recon image)
            real_img_sqz = real_img.squeeze(0)
            recon_img_sqz = recon_img.squeeze(0)
            writer.add_images(f'{experiment}-Epoch:{cur_epoch+1}|{epoch}/real pic & recon img-epoch:epoch_{cur_epoch+1}|{epoch},step:{i+1}|{num_batch}', torch.stack((real_img_sqz, recon_img_sqz), dim=0), cur_epoch * num_batch + i)
            # writer.add_image(f'real pic-step:{i+1}|{num_batch}', real_img, cur_epoch * num_batch + i)
            # writer.add_image(f'recon pic-step:{i+1}|{num_batch}', recon_img, cur_epoch * num_batch + i)

        # Save models checkpoints
        if i % save_minibatch_freq == save_minibatch_freq-1 or i == num_batch - 1:
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'{target}_encoder_epoch_{cur_epoch+1}|{epoch}_minibatch_{i+1}|{num_batch}.pt'))
            print(f"Save {target} encoder successfully - Epoch [{cur_epoch+1}/{epoch}], Step [{i+1}/{num_batch}]")

    
    # Update learning rates
    lr_scheduler.step()
