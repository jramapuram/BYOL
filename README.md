# SimCLR-pytorch

An implementation of [SimCLR](https://arxiv.org/abs/2002.05709) in pytorch.


## Usage Single GPU
**NOTE**: this will not produce good results. The paper by the authors use a batch size of 4096 for SOTA.
**NOTE**: Setup your github ssh tokens; if you get an authentication issue this is most likely it.


``` bash
> git clone --recursive git+ssh://git@github.com/jramapuram/SimCLR.git
# DATA_DIR is the location of imagenet or anything that works with imagefolder.
> python main.py --data-dir=$DATA_DIR --batch-size=64 --num-replicas=1 --epochs=100  # add --debug-step to do a single minibatch
```


## Usage SLURM

Setup stuff according to the [slurm bash script](./slurm/run.sh). Then:

``` bash
> sbatch slurm/run.sh
```
