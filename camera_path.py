from dataclasses import dataclass
import json
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import numpy as np

import torch

from transforms import Transforms

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
    reverted: bool

    def __init__(self, camera_path: dict, reverted=False):
        self.render_height = camera_path['render_height']
        self.render_width = camera_path['render_width']
        self.camera_orientations = [CameraOrientation(camera_orientation) for camera_orientation in camera_path['camera_path']]
        self.reverted = reverted
        if self.reverted:
            self.camera_orientations[::-1]
    
    def __getitem__(self, index: int) -> CameraOrientation:
        return self.camera_orientations[index]

    def __len__(self) -> int:
        return len(self.camera_orientations)


def generate_camera_path_dictionary_from_transforms(transforms_json_path: str, interpolation_steps=10) -> dict:
    with open(transforms_json_path, "r") as transforms_json:
        transforms = Transforms(json.load(transforms_json))
    rotations = []
    translations = []
    time_steps = [i for i in range(len(transforms.frames))]
    for frame in transforms.frames:
        rotation_matrix = R = frame.camera_extrinsics[0:3, 0:3].numpy()
        translation_vector = t = frame.camera_extrinsics[0:3, 3].numpy()
        rotation = Rotation.from_matrix(R)
        rotations.append(rotation)
        translations.append(t)
    rotations = Rotation.concatenate(rotations)
    rotation_slerp = Slerp(time_steps, rotations)
    translations_lerp = interp1d(time_steps, np.vstack(translations), axis=0)
    interpolation_times = np.linspace(0.0, len(transforms.frames) - 1, len(transforms.frames) * interpolation_steps)
    interpolated_rotations = rotation_slerp(interpolation_times)
    interpolated_translations = translations_lerp(interpolation_times)

    camera_path = []
    for R, t in zip(interpolated_rotations.as_matrix(), interpolated_translations):
        t = np.expand_dims(t, axis=0)
        homogeneous_camera_matrix = np.concatenate((np.concatenate((R, t.T), axis=1), [[0, 0, 0, 1]]), axis=0)
        camera_path.append({
            "camera_to_world": list(homogeneous_camera_matrix.flatten()),
            "fov": 55,
        })
    
    return {
        "render_height": 512,
        "render_width": 512,
        "camera_path": camera_path,
    }
    

if __name__ == "__main__":
    transforms_path = "/scratch/students/2023-spring-mt-mhnowak/text2room/final_output_2/final_experimenting_trajectory_3/full_trajectory/street_1/no_input_image_file/2023-08-15_10:20:06.348994Z/transforms_short.json"
    camera_path_dictionary = generate_camera_path_dictionary_from_transforms(transforms_path, interpolation_steps=10)
    with open("final_3_short_custom_camera_path.json", "w") as camera_path_json:
        json.dump(camera_path_dictionary , camera_path_json) 
