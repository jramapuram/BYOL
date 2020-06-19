#!/usr/bin/env python3

import torch


def regression_loss(x, y):
    """Pulled directly from BYOL"""
    norm_x, norm_y = x.norm(), y.norm()
    return -2 * torch.sum(x * y, dim=-1) / (norm_x * norm_y)


def loss_function(online_prediction1, online_prediction2, target_projection1, target_projection2):
    """BYOL loss.

    :param online_prediction1: the output of the final MLP of the online model for augmentation 1
    :param online_prediction2: the output of the final MLP of the online model for augmentation 2
    :param target_projection1: the output of the second-to-last MLP of the target model for augmentation 1
    :param target_projection2: the output of the second-to-last MLP of the target model for augmentation 1
    :returns: scalar loss
    :rtype: float32

    """
    loss_ab = regression_loss(online_prediction1, target_projection2.detach())
    loss_ba = regression_loss(online_prediction2, target_projection1.detach())
    return torch.mean(loss_ab + loss_ba)
