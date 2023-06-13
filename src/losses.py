import torch
from torch.nn.functional import binary_cross_entropy_with_logits


# define losses
def voxel_loss(voxel_src: torch.Tensor, voxel_tgt: torch.Tensor):
    loss = binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
    return loss


def chamfer_loss(point_cloud_src: torch.Tensor, point_cloud_tgt: torch.Tensor):
    assert point_cloud_src.ndimension() == 3  # (B, N, 3)
    assert point_cloud_src.size(-1) == 3
    assert point_cloud_tgt.ndimension() == 3  # (B, N, 3)
    assert point_cloud_tgt.size(-1) == 3

    # [B, N, N, 3]
    distance = point_cloud_src[:, :, None, :] - point_cloud_tgt[:, None, :, :]
    # [B, N, N]
    distance = torch.sum(distance**2, dim=-1)

    # [B, N]
    min_xy, _ = torch.min(distance, dim=-1)
    min_yx, _ = torch.min(distance.transpose(1, 2), dim=-1)

    # [B]
    loss_xy = min_xy.mean(dim=1)
    loss_yx = min_yx.mean(dim=1)

    return (loss_xy + loss_yx).mean()
