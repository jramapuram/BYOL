# SimCLR-pytorch

An implementation of [SimCLR](https://arxiv.org/abs/2002.05709) with DistributedDataParallel (1GPU : 1Process) in pytorch.  
This allows scalability to batch size of 4096 (suggested by authors) using 64 gpus each with batch size of 64.


## Usage Single GPU

**NOTE0**: this will not produce SOTA results, but is good for debugging. The authors use a batch size of 4096+ for SOTA results.  
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

  1. Start each replica worker pointing to the master using `--distributed-master=`
  2. Set the total number of replicas appropriately using `--num-replicas=`
  3. Set each node to have a unique `--distributed-rank=` ranging from `[0, num_replicas]`
  3. Ensure network connectivity between workers. You will get NCCL errors if there are resolution problems here.
  4. Profit
  
  
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
