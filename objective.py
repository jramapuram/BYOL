#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch.distributed as dist


def l2_normalize(x, dim=None, eps=1e-12):
    """Normalize a tensor over dim using the L2-norm."""
    sq_sum = torch.sum(torch.square(x), dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(torch.max(sq_sum, torch.ones_like(sq_sum)*eps))
    return x * inv_norm


def all_gather(tensor, expand_dim=0, num_replicas=None):
    """Gathers a tensor from other replicas, concat on expand_dim and return."""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    return torch.cat([o.unsqueeze(expand_dim) for o in other_replica_tensors], expand_dim)


def nt_xent(embedding1, embedding2, temperature=0.1, num_replicas=None):
    """NT-XENT Loss from SimCLR

    :param embedding1: embedding of augmentation1
    :param embedding2: embedding of augmentation2
    :param temperature: nce normalization temp
    :param num_replicas: number of compute devices
    :returns: scalar loss
    :rtype: float32

    """
    batch_size = embedding1.shape[0]
    feature_size = embedding1.shape[-1]
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    infinity = 1e9

    # normalize both embeddings
    embedding1 = l2_normalize(embedding1, dim=-1)
    embedding2 = l2_normalize(embedding2, dim=-1)

    if num_replicas > 1:
        # First grab the tensor from all other embeddings
        embedding1_full = all_gather(embedding1)
        embedding2_full = all_gather(embedding2)

        # fold the tensor in to create [B, F]
        embedding1_full = embedding1_full.reshape(-1, feature_size)
        embedding2_full = embedding2_full.reshape(-1, feature_size)

        # Create pseudo-labels using the current replica id & ont-hotting
        replica_id = dist.get_rank()
        labels = torch.arange(batch_size, device=embedding1.device) + replica_id * batch_size
        labels = labels.type(torch.int64)
        full_batch_size = embedding1_full.shape[0]
        cur_replica_labels = F.one_hot(labels, full_batch_size).to(embedding1_full.device)
        other_replica_labels = F.one_hot(labels, full_batch_size * 2).to(embedding1_full.device)
    else:
        embedding1_full = embedding1
        embedding2_full = embedding2
        cur_replica_labels = F.one_hot(torch.arange(batch_size), batch_size).to(embedding1.device)
        other_replica_labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(embedding1.device)

    masks = cur_replica_labels     # Mask out non-corresponding samples
    labels = other_replica_labels  # One-hot labels

    # Matmul-to-mask
    logits_aa = torch.matmul(embedding1, embedding1_full.T) / temperature
    logits_aa = logits_aa - masks * infinity
    logits_bb = torch.matmul(embedding2, embedding2_full.T) / temperature
    logits_bb = logits_bb - masks * infinity
    logits_ab = torch.matmul(embedding1, embedding2_full.T) / temperature
    logits_ba = torch.matmul(embedding2, embedding1_full.T) / temperature

    # Use our standard cross-entropy loss which uses log-softmax internally.
    # Concat on the feature dimension to provide all features for standard softmax-xent
    loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                             target=torch.argmax(labels, -1),
                             reduction="none")
    loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                             target=torch.argmax(labels, -1),
                             reduction="none")
    loss = loss_a + loss_b
    return torch.mean(loss)
