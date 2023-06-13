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
        self.gn0 = nn.GroupNorm(32, num_channels=latent_size * 2)
        self.fc1 = nn.Linear(latent_size * 2, latent_size * 4)
        self.gn1 = nn.GroupNorm(32, num_channels=latent_size * 4)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 512)
        # self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(latent_size * 4, self.num_points**3)
        # self.th = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = self.fc0(x)
        x = self.gn0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.gn1(x)
        x = F.relu(x)
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = self.th(self.fc5(x))
        x = x.view(-1, self.num_points, self.num_points, self.num_points)
        return x


class PointDecoder(nn.Module):
    def __init__(self, num_points: int, latent_size: int = 512):
        super().__init__()
        self.num_points = num_points
        out_features = self.num_points * 3

        layers = []
        scale = 2
        max_scale = out_features // latent_size
        prev_size = latent_size
        while scale < max_scale:
            next_size = latent_size * scale
            layers.extend(
                [
                    nn.Linear(prev_size, next_size),
                    nn.GroupNorm(32, num_channels=next_size),
                ]
            )
            prev_size = next_size
            scale *= 2

        layers.append(nn.Linear(prev_size, out_features))

        self.layers = nn.ModuleList(layers)
        # self.gn = nn.GroupNorm(32, num_channels=out_features)
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)
        # x = self.gn(x)
        x = self.th(x)
        x = x.view(-1, self.num_points, 3)
        return x


class SingleViewto3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dtype = cfg.dtype
        vision_model: torch.nn.Module = torchvision_resnet.__dict__[cfg.arch](
            pretrained=True
        )
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)

        latent_size = 512
        for module in vision_model.modules():
            if isinstance(module, torchvision_resnet.BasicBlock):
                latent_size *= torchvision_resnet.BasicBlock.expansion
                break
            elif isinstance(module, torchvision_resnet.Bottleneck):
                latent_size *= torchvision_resnet.Bottleneck.expansion
                break

        # define decoder
        if cfg.dtype == "voxel":
            self.decoder = VoxelDecoder(cfg.n_points, latent_size)
        elif cfg.dtype == "point":
            self.decoder = PointDecoder(cfg.n_points, latent_size)

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, images: torch.Tensor):
        images = self.normalize(images)
        encoded_feat = self.encoder(images)
        encoded_feat = torch.flatten(encoded_feat, 1)
        pred = logits = self.decoder(encoded_feat)
        if self.dtype == "voxel":
            pred = torch.sigmoid(logits)
        return logits, pred
