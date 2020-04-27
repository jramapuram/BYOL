# SimCLR-pytorch

An implementation of [SimCLR](https://arxiv.org/abs/2002.05709) in pytorch.


## Usage

**NOTE**: Setup your github ssh tokens; if you get an authentication issue this is most likely it.

``` bash
> git clone --recursive git+ssh://git@github.com/jramapuram/SimCLR.git
# DATA_DIR is the location of imagenet or anything that works with imagefolder.
> python main.py --data-dir=$DATA_DIR --batch-size=4096 --num-replicas=8  # trains with 8 GPUS locally
```
