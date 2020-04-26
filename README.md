# SimCLR-pytorch

An implementation of [SimCLR](https://arxiv.org/abs/2002.05709) in pytorch.


## Usage

Setup your github ssh tokens; if you get an authentication issue this is most likely it.
This model trains on 8x GPUs, YMMV on your custom setup.

``` bash
git clone --recursive git+ssh://git@github.com/jramapuram/SimCLR.git
python main.py --data-dir=$DATA_DIR  # DATA_DIR is the location of imagenet.
```
