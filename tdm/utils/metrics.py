import torch
import torch.nn.functional as F


def dice_score(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               n_categories=1,
               smooth=0.,
               eps=1e-7):
    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)
    if n_categories > 1:
        y_true = F.one_hot(y_true, n_categories+1).permute(0, 2, 1)[:, 1:, :]
        y_pred = F.one_hot(y_pred, n_categories+1).permute(0, 2, 1)[:, 1:, :]
    assert y_true.shape == y_pred.shape, "Shape mismatch"
    intersection = torch.sum(y_true * y_pred, dim=-1)
    total_area = torch.sum(y_true + y_pred, dim=-1)

    return torch.mean(
        (2 * intersection + smooth) / (total_area + smooth).clamp(min=eps))
