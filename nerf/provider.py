import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from einops import (rearrange, reduce, repeat)

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from camera_path import CameraPath

from .utils import get_rays, get_variable_resolution_rays, safe_normalize
from transforms import Transforms

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (right) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (left) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, opt, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                torch.randn(size, device=device),
                torch.abs(torch.randn(size, device=device)),
                torch.randn(size, device=device),
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if opt.jitter_pose:
        jit_center = opt.jitter_center # 0.015  # was 0.2
        jit_target = opt.jitter_target
        centers += torch.rand_like(centers) * jit_center - jit_center/2.0
        targets += torch.randn_like(centers) * jit_target

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if opt.jitter_pose:
        up_noise = torch.randn_like(up_vector) * opt.jitter_up
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), return_dirs=False, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000 # infinite

        # [debug] visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, opt, radius_range=self.opt.radius_range, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def get_default_view_data(self):

        H = int(self.opt.known_view_scale * self.H)
        W = int(self.opt.known_view_scale * self.W)
        cx = H / 2
        cy = W / 2

        radii = torch.FloatTensor(self.opt.ref_radii).to(self.device)
        thetas = torch.FloatTensor(self.opt.ref_polars).to(self.device)
        phis = torch.FloatTensor(self.opt.ref_azimuths).to(self.device)
        poses, dirs = circle_poses(self.device, radius=radii, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
        fov = self.opt.default_fovy
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, cx, cy])

        projection = torch.tensor([
            [2*focal/W, 0, 0, 0],
            [0, -2*focal/H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(radii), 1, 1)

        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, H, W, -1)

        data = {
            'H': H,
            'W': W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': self.opt.ref_polars,
            'azimuth': self.opt.ref_azimuths,
            'radius': self.opt.ref_radii,
        }

        return data

    def collate(self, index):
        # ! This gets called when creating the data loader 

        B = len(index)

        if self.training:
            # random pose on the fly
            poses, dirs, thetas, phis, radius = rand_poses(B, self.device, self.opt, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]

        elif self.type == 'six_views':
            # six views
            thetas_six = [90]*4 + [1e-6] + [180]
            phis_six = [0, 90, 180, -90, 0, 0]
            thetas = torch.FloatTensor([thetas_six[index[0]]]).to(self.device)
            phis = torch.FloatTensor([phis_six[index[0]]]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = self.opt.default_fovy

        else:
            # circle pose
            thetas = torch.FloatTensor([self.opt.default_polar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)

            # fixed focal
            fov = self.opt.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader


class ClassicNeRFDataset(torch.utils.data.Dataset):
    def __init__(self, opt, device, transforms_json_path: str, type='train'):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.training = self.type in ['train', 'all']

        self.near = self.opt.min_near
        self.far = 1000 # infinite

        with open(transforms_json_path, "r") as transforms_json:
            transforms_directory = os.path.dirname(transforms_json_path)
            self.transforms = Transforms(json.load(transforms_json))
            self.image_paths = [os.path.join(transforms_directory, frame.file_path) for frame in self.transforms.frames]
            self.depth_paths = [os.path.join(transforms_directory, frame.depth_file_path) for frame in self.transforms.frames]

        self.size = len(self.transforms.frames)

        images = [cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image_path in self.image_paths]
        rgba = np.stack([image.astype(np.float32) / 255 for image in images])
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb = torch.from_numpy(rgb).permute(0,3,1,2).contiguous().to(self.device)
        self.mask = torch.from_numpy(rgba[..., 3] > 0.5).to(self.device)

        depths = [np.load(depth_path) for depth_path in self.depth_paths]
        depth = np.stack(depths)
        self.depth = torch.from_numpy(depth).to(self.device)

        # [debug] visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, opt, radius_range=self.opt.radius_range, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def get_default_view_data(self):
        return self.__getitem__(0)
    
    def __getitem__(self, index: int):
        source_image_height = int(self.transforms.height)
        source_image_width = int(self.transforms.width)
        render_image_height = int(self.opt.known_view_scale * self.transforms.height)
        render_image_width = int(self.opt.known_view_scale * self.transforms.width)

        focal_length_x = self.transforms.focal_length_x
        focal_length_y = self.transforms.focal_length_y
        principal_point_x = self.transforms.principal_point_x
        principal_point_y = self.transforms.principal_point_y

        poses = torch.stack([frame.camera_extrinsics for frame in self.transforms.frames]).to(self.device)
        intrinsics = np.array([focal_length_x, focal_length_y, principal_point_x, principal_point_y])

        projection = torch.tensor(
            [
                [2*focal_length_x/source_image_width, 0, 0, 0],
                [0, -2*focal_length_y/source_image_height, 0, 0],
                [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                [0, 0, -1, 0]
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(len(self.transforms.frames), 1, 1)

        mvp = projection @ torch.inverse(poses) # [B, 4, 4]

        # sample a low-resolution but full image
        number_of_rays = render_image_height * render_image_width
        rays_o, rays_d, inds = get_variable_resolution_rays(
            poses,
            intrinsics,
            source_image_height,
            source_image_width,
            render_image_height,
            render_image_width
        )

        rgb = rearrange(self.rgb[index], 'C H W-> (H W) C')[inds[index]].reshape(render_image_height, render_image_width, 3).permute(2, 0, 1).contiguous() # [3, H, W]
        depth = self.depth[index].flatten()[inds[index]].reshape(1, render_image_height, render_image_width).contiguous() # [1, H, W]

        data = {
            'H': render_image_height,
            'W': render_image_width,
            'rays_o': rays_o[index], # [N, 3]
            'rays_d': rays_d[index], # [N, 3]
            'mvp': mvp[index], # [4, 4]
            'rgb': rgb,
            'depth': depth,
        }

        return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(self, batch_size=batch_size, shuffle=self.training, num_workers=0)
        loader._data = self
        return loader
    
    def __len__(self):
        return self.rgb.size(0)


class NeRFRenderingDataset:
    def __init__(self, opt, device, camera_path: CameraPath):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.camera_path = camera_path

        self.H = self.camera_path.render_height
        self.W = self.camera_path.render_width
        self.size = len(camera_path)

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.opt.min_near
        self.far = 1000 # infinite

    def collate(self, index: list[int]):
        fov = self.camera_path[index[0]].field_of_view

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        poses = torch.stack([self.camera_path[i].camera_extrinsics for i in index]).to(self.device)

        projection = torch.tensor([
            [2*focal/self.W, 0, 0, 0],
            [0, -2*focal/self.H, 0, 0],
            [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp': mvp,
        }

        return data

    def dataloader(self, batch_size=1):
        batch_size = batch_size or self.opt.batch_size
        loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=False, num_workers=0)
        loader._data = self
        return loader
