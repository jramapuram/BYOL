# ml_base : A base to startup your VAE or classifier/regressor projects

ml_base is intended to be used as a starting point for quick prototyping of Variational Autoencoder or classifier / regressor projects


## Usage Classifier/Regressor Project

First go create your repository to house your project (`github.com/YOUR_USERNAME/YOUR_PROJECT.git` below).

``` bash
git clone --recursive git+ssh://git@github.com/jramapuram/ml_base.git                # clone the repo
git remote set-url origin git+ssh://git@github.com/YOUR_USERNAME/YOUR_PROJECT.git    # change the endpoint

# Prototype the idea you want

git push -f                                                                          # push to new remote
```

#### Example resnet18 classifier project

Change the transforms in `supervised_main.py` appropriately for cifar10 and:

``` bash
python supervised_main.py --task=cifar10
```

More complex example:

``` bash
python supervised_main.py --lr-update-schedule=cosine --warmup=10 --optimizer=lars_adam \  
--lr=1e-3 --task=imagefolder --data-dir=$HOME/datasets/imagenet --batch-size=64 \  
--visdom-url=http://localhost --visdom-port=8097 --epochs=50 --polyak-ema=0 \  
--weight-decay=1e-6 --model-dir=.models --arch=resnet50 --uid=supv00_0
```


## Usage VAE Project

First go create your repository to house your project (`github.com/YOUR_USERNAME/YOUR_PROJECT.git` below).

``` bash
git clone --recursive git+ssh://git@github.com/jramapuram/ml_base.git                # clone the repo
git clone --recursive git+ssh://git@github.com/jramapuram/vae.git models/vae         # clone the VAE repo (consider sub-moduling)
git remote set-url origin git+ssh://git@github.com/YOUR_USERNAME/YOUR_PROJECT.git    # change the endpoint

# Prototype the idea you want

git push -f                                                                          # push to new remote
```


#### Example Convolutional VAE Usage

``` bash
python vae_main.py --vae-type=simple --debug-step --disable-gated --reparam-type=isotropic_gaussian
```


#### Example VRNN Usage

``` bash
python vae_main.py --vae-type=vrnn --debug-step --disable-gated --reparam-type=isotropic_gaussian
```


## Sphinx Documentation Generator

``` bash
(base) ➜  ml_base git:(master) ✗ sphinx-quickstart
Welcome to the Sphinx 2.0.1 quickstart utility.

Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets).

Selected root path: .

You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]:

The project name will occur in several places in the built documentation.
> Project name: ml_base
> Author name(s): Jason Ramapuram
> Project release []: 0.1

If the documents are to be written in a language other than English,
you can select a language here by its language code. Sphinx will then
translate text that it generates into that language.

For a list of supported codes, see
http://sphinx-doc.org/config.html#confval-language.
> Project language [en]:

Creating file ./conf.py.
Creating file ./index.rst.
Creating file ./Makefile.
Creating file ./make.bat.

Finished: An initial directory structure has been created.

You should now populate your master file ./index.rst and create other documentation
source files. Use the Makefile to build the docs, like so:
   make builder
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.

# Then follow https://pythonhosted.org/an_example_pypi_project/sphinx.html
```
