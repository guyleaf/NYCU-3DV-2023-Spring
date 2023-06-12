import logging
import os
import time

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from pytorch3d.ops import cubify, estimate_pointcloud_normals
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Pointclouds

import src.losses as losses
from src.dataset import ShapeNetDB
from src.losses import ChamferDistanceLoss
from src.model import SingleViewto3D
from src.visualize import PointScene, VoxelScene

# A logger for this file
log = logging.getLogger(__name__)

cd_loss = ChamferDistanceLoss()


def calculate_loss(predictions, ground_truth, cfg):
    if cfg.dtype == "voxel":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif cfg.dtype == "point":
        loss = cd_loss(predictions, ground_truth)
    # elif cfg.dtype == 'mesh':
    #     sample_trg = sample_points_from_meshes(ground_truth, cfg.n_points)
    #     sample_pred = sample_points_from_meshes(predictions, cfg.n_points)

    #     loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
    #     loss_smooth = losses.smoothness_loss(predictions)

    # loss = cfg.w_chamfer * loss_reg + cfg.w_smooth * loss_smooth
    return loss


def plot_points(predictions: torch.Tensor) -> torch.Tensor:
    normals = estimate_pointcloud_normals(predictions)
    point_clouds = Pointclouds(
        points=predictions,
        normals=normals,
        features=torch.full_like(predictions, 0.5, device=predictions.device),
    )

    scene = PointScene(predictions.device)
    scene.set_cam(1.8, 45.0, 45.0)
    scene.set_rasterizer(image_size=256)
    scene.set_renderer()
    images = scene.renderer(point_clouds)
    return images


def plot_voxels(predictions: torch.Tensor):
    meshes = cubify(predictions, thresh=0.5, align="center")

    verts_list = meshes.verts_packed()
    verts_rgb = torch.ones(1, verts_list.shape[0], 3, device=predictions.device)
    meshes.textures = TexturesVertex(verts_features=verts_rgb)

    scene = VoxelScene(predictions.device)
    scene.set_cam(2.5, 45.0, 45.0)
    scene.set_light(location=[[0.0, 0.0, 0.0]])
    scene.set_rasterizer(image_size=256)
    scene.set_renderer()
    images = scene.renderer(meshes)
    return images


@torch.no_grad()
@hydra.main(config_path="configs/", config_name="config.yaml")
def evaluate_model(cfg: DictConfig):
    device = torch.device(cfg.device)

    shapenetdb = ShapeNetDB(cfg.data_dir, cfg.dtype, cfg.n_points)

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    eval_loader = iter(loader)

    if cfg.dtype == "voxel":
        cfg.n_points = shapenetdb[0][1].shape[1]

    model = SingleViewto3D(cfg)
    model.cuda(device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    avg_loss = []

    if cfg.load_eval_checkpoint:
        checkpoint = torch.load(f"{cfg.base_dir}/checkpoint_{cfg.dtype}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(f"Succesfully loaded iter {start_iter}")

    vis_dir = os.path.join(cfg.base_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    log.info("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        images_gt, ground_truth_3d, _ = next(eval_loader)
        images_gt, ground_truth_3d = images_gt.cuda(device), ground_truth_3d.cuda(
            device
        )

        read_time = time.time() - read_start_time

        prediction_3d: torch.Tensor = model(images_gt)
        torch.save(prediction_3d.cpu(), f"{cfg.base_dir}/pre_point_cloud.pt")

        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg).cpu().item()

        if (step % cfg.vis_freq) == 0:
            # visualization block
            if cfg.dtype == "point":
                gt_images = plot_points(ground_truth_3d)
                images = plot_points(prediction_3d)
            elif cfg.dtype == "voxel":
                gt_images = plot_voxels(ground_truth_3d)
                images = plot_voxels(prediction_3d)

            rgb_images = images_gt.permute(0, 2, 3, 1).cpu().numpy()
            gt_images = gt_images.cpu().numpy()
            images = images.cpu().numpy()

            f = plt.figure()
            for i in range(images.shape[0]):
                rgb_image = rgb_images[i]
                gt_image = gt_images[i]
                image = images[i]

                axeses = f.subplots(1, 3, sharey=True)
                for axes in axeses:
                    axes.xaxis.set_visible(False)
                    axes.yaxis.set_visible(False)

                ax1, ax2, ax3 = axeses
                ax1.set_title("GT Image")
                ax1.imshow(rgb_image)

                ax2.set_title("GT 3D")
                ax2.imshow(gt_image)

                ax3.set_title("Prediction 3D")
                ax3.imshow(image)

                file_name = os.path.join(vis_dir, f"{step}_{cfg.dtype}_{i}.png")
                f.savefig(file_name)
                f.clf()
            plt.close(f)

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        avg_loss.append(loss)

        log.info(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); eva_loss: %.3f"
            % (
                step,
                max_iter,
                total_time,
                read_time,
                iter_time,
                torch.tensor(avg_loss).mean(),
            )
        )

    log.info("Done!")


if __name__ == "__main__":
    evaluate_model()
