import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as torchvision_resnet
from torchvision.transforms import Normalize

# from pytorch3d.utils import ico_sphere
# import pytorch3d


class VoxelDecoder(nn.Module):
    def __init__(self, num_points: int, latent_size: int = 512):
        super().__init__()
        self.num_points = num_points
        self.fc0 = nn.Linear(latent_size, latent_size * 2)
        self.fc1 = nn.Linear(latent_size * 2, latent_size * 4)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(latent_size * 4, self.num_points**3)
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(-1, self.num_points, self.num_points, self.num_points)
        return x


class PointDecoder(nn.Module):
    def __init__(self, num_points: int, latent_size: int = 512):
        super().__init__()
        self.num_points = num_points

        layers = []
        scale = 2
        max_scale = (self.num_points * 3) // latent_size
        prev_size = latent_size
        while scale < max_scale:
            next_size = latent_size * scale
            layers.append(nn.Linear(prev_size, next_size))
            prev_size = next_size
            scale *= 2

        layers.append(nn.Linear(prev_size, self.num_points * 3))

        self.layers = nn.ModuleList(layers)
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.th(self.layers[-1](x))
        x = x.view(-1, self.num_points, 3)
        return x


class SingleViewto3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vision_model: torch.nn.Module = torchvision_resnet.__dict__[cfg.arch](
            pretrained=True
        )
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))

        latent_size = 512
        for module in vision_model.modules():
            if isinstance(module, torchvision_resnet.BasicBlock):
                latent_size *= torchvision_resnet.BasicBlock.expansion
            elif isinstance(module, torchvision_resnet.Bottleneck):
                latent_size *= torchvision_resnet.Bottleneck.expansion

        # define decoder
        if cfg.dtype == "voxel":
            self.decoder = VoxelDecoder(cfg.n_points, latent_size)
        elif cfg.dtype == "point":
            self.decoder = PointDecoder(cfg.n_points, latent_size)
        # elif cfg.dtype == "mesh":
        #     # try different mesh initializations
        #     mesh_pred = ico_sphere(4,'cuda')
        #     self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*cfg.batch_size, mesh_pred.faces_list()*cfg.batch_size)
        #     # TODO:
        #     # self.decoder =

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, images: torch.Tensor):
        images = self.normalize(images)
        encoded_feat = self.encoder(images)
        encoded_feat = torch.flatten(encoded_feat, 1)
        pred = self.decoder(encoded_feat)
        return pred

        # elif cfg.dtype == "mesh":
        #     # TODO:
        #     # deform_vertices_pred =
        #     mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
        #     return  mesh_pred
