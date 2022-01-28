import torch
import torch.nn.functional as F


def _soft_dice_score(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     smooth=0.,
                     eps=1e-7):
    assert y_true.shape == y_pred.shape, "Shape mismatch"
    intersection = torch.sum(y_true * y_pred, dim=-1)
    total_area = torch.sum(y_true + y_pred, dim=-1)
    score = torch.mean(
        (2 * intersection + smooth) / (total_area + smooth).clamp(min=eps))

    return score


def _soft_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps=1e-7):
    assert y_true.shape == y_pred.shape, "Shape mismatch"
    intersection = torch.sum(y_true * y_pred, dim=-1)
    union = torch.sum(y_true + y_pred, dim=-1) - intersection
    score = torch.mean(intersection / union.clamp(min=eps))

    return score


def _preprocess_inputs(y_true: torch.Tensor,
                       y_pred: torch.Tensor,
                       from_logits=True,
                       n_categories=1):
    assert y_pred.shape[0] == y_true.shape[0]
    N = y_true.shape[0]
    y_true = y_true.view(N, -1)
    if n_categories == 1:
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.view(N, -1)
    else:
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.view(N, n_categories + 1, -1)[:, 1:, :]
        else:
            y_pred = y_pred.view(N, -1)
            y_pred = F.one_hot(y_pred, n_categories + 1)
            y_pred = y_pred.permute(0, 2, 1)[:, 1:, :]
        y_true = F.one_hot(y_true, n_categories + 1)
        y_true = y_true.permute(0, 2, 1)[:, 1:, :]
    
    return y_true, y_pred


def dice_score(y_true: torch.Tensor,
               y_pred: torch.Tensor,
               from_logits=True,
               n_categories=1,
               smooth=0.,
               eps=1e-7):
    y_true, y_pred = _preprocess_inputs(y_true, y_pred, from_logits, n_categories)

    return _soft_dice_score(y_true, y_pred, smooth, eps)


def iou_score(y_true: torch.Tensor,
              y_pred: torch.Tensor,
              from_logits=True,
              n_categories=1,
              eps=1e-7):
    y_true, y_pred = _preprocess_inputs(y_true, y_pred, from_logits, n_categories)

    return _soft_iou_score(y_true, y_pred, eps)