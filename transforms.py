from dataclasses import dataclass

import torch

@dataclass
class Frame:
    camera_extrinsics: torch.Tensor
    file_path: str
    depth_file_path: str

    def __init__(self, frame: dict):
        self.camera_extrinsics = torch.Tensor(frame['transform_matrix'])
        self.file_path = frame['file_path']
        self.depth_file_path = frame['depth_file_path']

@dataclass
class Transforms:
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    height: int
    width: int
    integer_depth_scale: int
    frames: list[Frame]

    def __init__(self, transforms: dict):
        self.focal_length_x = transforms['fl_x']
        self.focal_length_y = transforms['fl_y']
        self.principal_point_x = transforms['cx']
        self.principal_point_y = transforms['cy']
        self.height = transforms['h']
        self.width = transforms['w']
        self.integer_depth_scale = transforms['integer_depth_scale']
        self.frames = [Frame(frame) for frame in transforms['frames']]
