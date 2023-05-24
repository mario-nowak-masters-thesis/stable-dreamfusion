from dataclasses import dataclass

import torch

@dataclass
class CameraOrientation:
    camera_extrinsics: torch.Tensor
    field_of_view: int

    def __init__(self, camera_orientation: dict):
        self.camera_extrinsics = torch.Tensor(camera_orientation['camera_to_world']).reshape(4, 4)
        self.field_of_view = camera_orientation['fov']

@dataclass
class CameraPath:
    render_height: int
    render_width: int
    camera_orientations: list[CameraOrientation]

    def __init__(self, camera_path: dict):
        self.render_height = camera_path['render_height']
        self.render_width = camera_path['render_width']
        self.camera_orientations = [CameraOrientation(camera_orientation) for camera_orientation in camera_path['camera_path']]
    
    def __getitem__(self, index: int) -> CameraOrientation:
        return self.camera_orientations[index]

    def __len__(self) -> int:
        return len(self.camera_orientations)
