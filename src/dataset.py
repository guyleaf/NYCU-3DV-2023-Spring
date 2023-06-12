import argparse
import glob

# from PIL import Image
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# from torchvision import datasets
from torchvision.io import read_image

# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj
# from pytorch3d.renderer import TexturesVertex


class ShapeNetDB(Dataset):
    def __init__(self, data_dir: str, data_type: str, n_points: int = 2048):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.n_points = n_points
        self.db = self.load_db()

        self.get_index()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        if self.data_type == "point":
            """
            Return shapes:
            img: (B, 256, 256, 3)
            pc: (B, 2048, 3)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            pc, object_id = self.load_point(idx)

            assert img_id == object_id

            return img, pc, object_id

        elif self.data_type == "voxel":
            """
            Return shapes:
            img: (B, 256, 256, 3)
            voxel: (B, 33, 33, 33)
            object_id: (B,)
            """
            img, img_id = self.load_img(idx)
            voxel, object_id = self.load_voxel(idx)

            assert img_id == object_id

            return img, voxel, object_id

        # elif self.data_type == 'mesh':
        #     img, img_id = self.load_img(idx)
        #     mesh, object_id = self.load_mesh(idx)

        #     assert img_id == object_id

        #     return img, mesh, object_id

    def load_db(self):
        # print(os.path.join(self.data_dir, '*'))
        db_list = sorted(glob.glob(os.path.join(self.data_dir, "*")))
        # print(db_list)

        return db_list

    def get_index(self):
        self.id_index = self.data_dir.split("/").index("data") + 2
        # print(self.id_index)

    def load_img(self, idx):
        path = os.path.join(self.db[idx], "view.png")
        img = read_image(path) / 255.0

        object_id = self.db[idx].split("/")[self.id_index]

        return img, object_id

    # def load_mesh(self, idx):
    #     path = os.path.join(self.db[idx], 'model.obj')
    #     verts, faces, _ = load_obj(path, load_textures=False)
    #     faces_idx = faces.verts_idx

    #     # normalize
    #     center = verts.mean(0)
    #     verts = verts - center
    #     scale = max(verts.abs().max(0)[0])
    #     verts = verts / scale

    # make white texturre
    # verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # textures = TexturesVertex(verts_features=verts_rgb)

    # mesh = Meshes(
    #     verts=[verts],
    #     faces=[faces_idx],
    #     textures=textures
    # )

    # object_id = self.db[idx].split('/')[self.id_index]

    # return mesh, object_id

    def load_point(self, idx):
        path = os.path.join(self.db[idx], "point_cloud.npy")
        points: np.ndarray = np.load(path)

        # resample
        if self.n_points < points.shape[0]:
            choice = np.random.choice(points.shape[0], self.n_points, replace=False)
            points = points[choice]

        # normalize
        points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
        points = points / dist  # scale

        object_id = self.db[idx].split("/")[self.id_index]

        return torch.from_numpy(points), object_id

    def load_voxel(self, idx):
        path = os.path.join(self.db[idx], "voxel.npy")
        voxel = np.load(path)

        object_id = self.db[idx].split("/")[self.id_index]

        return torch.from_numpy(voxel).float(), object_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="The root path of dataset")
    parser.add_argument(
        "--dtype", default="point", type=str, help="The format of dataset"
    )
    parser.add_argument(
        "--n_points", default=2048, type=int, help="The number of points"
    )
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    # from pytorch3d.datasets import collate_batched_meshes

    db = ShapeNetDB(args.root_dir, args.dtype, args.n_points)
    dataloader = DataLoader(db, batch_size=10, shuffle=True)

    for img, point, object_id in dataloader:
        print(img.shape)
        print(point.shape)
        print(object_id)
        break
