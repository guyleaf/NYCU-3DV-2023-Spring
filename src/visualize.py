import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    NormWeightedCompositor,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftGouraudShader,
    look_at_view_transform,
)


class VoxelScene:
    def __init__(self, device: torch.device):
        self.device = device

    def set_cam(self, dist=1.0, elev=0.0, azim=0.0):
        """
        Initialize a camera.
        With world coordinates +Y up, +X left and +Z in.
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_view_transform
        """
        R, T = look_at_view_transform(dist, elev, azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        return self

    def set_light(self, location=[[0.0, 0.0, 0.0]]):
        """
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/lighting.html#pytorch3d.renderer.lighting.PointLights
        """
        self.lights = PointLights(location=location, device=self.device)
        return self

    def set_rasterizer(self, image_size=512):
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-5,
            faces_per_pixel=150,
            bin_size=0,
            cull_backfaces=True,
            # max_faces_per_bin=10,
        )
        """
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/rasterizer.html#pytorch3d.renderer.mesh.rasterizer.RasterizationSettings
        """
        return self

    def set_renderer(self):
        """
        see full api in https://github.com/facebookresearch/pytorch3d/blob/2c64635daa2aa728f35ed4abe41c6942ae8c0d8b/pytorch3d/renderer/mesh/renderer.py#L32
        """

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings
            ),
            shader=SoftGouraudShader(
                device=self.device, cameras=self.cameras, lights=self.lights
            ),
        )

        return self


class PointScene:
    def __init__(self, device: torch.device):
        self.device = device

    def set_cam(self, dist=1.0, elev=0.0, azim=0.0):
        """
        Initialize a camera.
        With world coordinates +Y up, +X left and +Z in.
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html#pytorch3d.renderer.cameras.look_at_view_transform
        """
        R, T = look_at_view_transform(dist, elev, azim)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        return self

    def set_rasterizer(self, image_size=512):
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=1e-2,
            points_per_pixel=120,
            bin_size=0,
            # max_points_per_bin=10,
        )
        """
        see full api in https://pytorch3d.readthedocs.io/en/latest/modules/renderer/rasterizer.html#pytorch3d.renderer.mesh.rasterizer.RasterizationSettings
        """
        return self

    def set_renderer(self):
        """
        see full api in https://github.com/facebookresearch/pytorch3d/blob/2c64635daa2aa728f35ed4abe41c6942ae8c0d8b/pytorch3d/renderer/mesh/renderer.py#L32
        """

        self.renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings
            ),
            compositor=NormWeightedCompositor(background_color=[1.0, 1.0, 1.0]),
        )

        return self
