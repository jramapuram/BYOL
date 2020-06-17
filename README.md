# SimCLR-pytorch

An implementation of [SimCLR](https://arxiv.org/abs/2002.05709) with DistributedDataParallel (1GPU : 1Process) in pytorch.  
This allows scalability to batch size of 4096 (suggested by authors) using 64 gpus, each with batch size of 64 at a resolution of 224x224x3 in FP32 (see below for FP16 support).


## Usage Single GPU

**NOTE0**: this will not produce SOTA results, but is good for debugging. The authors use a batch size of 4096+ for SOTA.    
**NOTE1**: Setup your github ssh tokens; if you get an authentication issue from the git clone this is most likely it.


``` bash
> git clone --recursive git+ssh://git@github.com/jramapuram/SimCLR.git
# DATADIR is the location of imagenet or anything that works with imagefolder.
> ./docker/run.sh "python main.py --data-dir=$DATADIR \  
                                  --batch-size=64 \  
                                  --num-replicas=1 \  
                                  --epochs=100" 0  # add --debug-step to do a single minibatch
```
The bash script `docker/run.sh` pulls the appropriate docker container.  
If you want to setup your own environment use:
  - `environment.yml` (conda) in **addition** to
  - `requirements.txt` (pip)  
  
or just take a look at the Dockerfile in `docker/Dockerfile`.

## Usage SLURM

Setup stuff according to the [slurm bash script](./slurm/run.sh). Then:

``` bash
> cd slurm && sbatch run.sh
```


## Usage custom cluster / AWS, etc

  1. Start each replica worker pointing to the master using `--distributed-master=`.
  2. Set the total number of replicas appropriately using `--num-replicas=`.
  3. Set each node to have a unique `--distributed-rank=` ranging from `[0, num_replicas)`.
  3. Ensure network connectivity between workers. You will get NCCL errors if there are resolution problems here.
  4. Profit.
  
For example, with a 2 node setup run the following on the master node:
```bash
python main.py \
     --epochs=100 \
     --data-dir=<YOUR_DATA_DIR> \
     --batch-size=128 \                   # divides into 64 per node
     --convert-to-sync-bn \
     --visdom-url=http://MY_VISDOM_URL \  # optional, not providing uses tensorboard
     --visdom-port=8097 \                 # optional, not providing uses tensorboard
     --num-replicas=2 \                   # specifies total available nodes, 2 in this example     
     --distributed-master=127.0.0.1 \
     --distributed-port=29301 \
     --distributed-rank=0 \               # rank-0 is the master
     --uid=simclrv00_0
```

and the following on the child node:

```bash
export MASTER=<IP_ADDR_OF_MASTER_ABOVE>
python main.py \
     --epochs=100 \
     --data-dir=<YOUR_DATA_DIR> \
     --batch-size=128 \                   # divides into 64 per node
     --convert-to-sync-bn \
     --visdom-url=http://MY_VISDOM_URL \  # optional, not providing uses tensorboard
     --visdom-port=8097 \                 # optional, not providing uses tensorboard
     --num-replicas=2 \                   # specifies total available nodes, 2 in this example
     --distributed-master=$MASTER \
     --distributed-port=29301 \
     --distributed-rank=1 \               # rank-1 is this child, increment for extra nodes
     --uid=simclrv00_0
```


## Setup data

Grab imagenet, [do standard pre-processing](https://github.com/soumith/imagenet-multiGPU.torch#data-processing) and use `--data-dir=${DATA_DIR}`. **Note:** This SimCLR implementation expects two pytorch `imagefolder` locations: `train` and `test` as opposed to `val` in the preprocessor above.

## FP16 support

If you have GPUs that works well with FP16, you can try the `--half` flag.  
This will allow faster training with larger batch sizes (~95 with a 12Gb GPU memory).  
If training doesn't work well try chaning the [AMP optimization](https://nvidia.github.io/apex/amp.html#opt-levels) level [here](https://github.com/jramapuram/SimCLR/blob/master/main.py#L590).

## IO bound / Slow data processing?

Try increasing `--workers-per-replica` for dataloading or placing your dataset on a drive with larger IOPS.  
Optionally, you can also try to use the [Nvidia DALI](https://github.com/NVIDIA/DALI) image loading backend by specifying `--task=dali_multi_augment_image_folder`. However, the latter is missing the grayscale and gaussian blur augmentations, so model  performance might be degraded.

## Visualize results

This implementation supports tensorboard and visdom.  
Omitting the `--visdom-url` and `--visdom-port` args defaults to tensorboard (which stores in `./runs`).
  
## Citation

Cite the original authors on doing some great work:

```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
```

Like this replication? Buy me [a beer](https://github.com/sponsors/jramapuram).
