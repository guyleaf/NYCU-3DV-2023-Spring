import logging
import sys
import time

import hydra
import torch
from omegaconf import DictConfig

import src.losses as losses
from src.dataset import ShapeNetDB
from src.losses import ChamferDistanceLoss
from src.model import SingleViewto3D

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


@hydra.main(config_path="configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    log.info(f"Device: {cfg.device}")
    device = torch.device(cfg.device)

    log.info(cfg.data_dir)
    shapenetdb = ShapeNetDB(cfg.data_dir, cfg.dtype)

    loader = torch.utils.data.DataLoader(
        shapenetdb,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )
    train_loader = iter(loader)

    if cfg.dtype == "voxel":
        cfg.n_points = shapenetdb[0][1].shape[1]

    model = SingleViewto3D(cfg)
    model.cuda(device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if cfg.load_checkpoint:
        checkpoint = torch.load(f"{cfg.base_dir}/checkpoint_{cfg.dtype}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        log.info(f"Succesfully loaded iter {start_iter}")

    log.info("Starting training !")
    best_loss = sys.float_info.max
    for step in range(start_iter, cfg.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        images_gt, ground_truth_3d, _ = next(train_loader)
        images_gt, ground_truth_3d = images_gt.cuda(device), ground_truth_3d.cuda(
            device
        )

        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt)

        loss = calculate_loss(prediction_3d, ground_truth_3d, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.detach().cpu().item()

        if loss_vis < best_loss:
            best_loss = loss_vis
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{cfg.base_dir}/checkpoint_{cfg.dtype}.pth",
            )

        # if (step % cfg.save_freq) == 0:
        #     torch.save(
        #         {
        #             "step": step,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #         },
        #         f"{cfg.base_dir}/checkpoint_{cfg.dtype}.pth",
        #     )

        log.info(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.5f"
            % (step, cfg.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    log.info("Done!")


if __name__ == "__main__":
    train_model()
