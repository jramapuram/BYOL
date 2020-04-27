import os
import time
import argparse
import functools
import pprint
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np


from torchvision import transforms
import torchvision.models as models

from objective import nt_xent
import helpers.metrics as metrics
import helpers.layers as layers
import helpers.utils as utils
import optimizers.scheduler as scheduler

from datasets.loader import get_loader
from helpers.grapher import Grapher
from optimizers.lars import LARS


# Grab all the model names from torchvision
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='SimCLR Pytorch')

# Task parameters
parser.add_argument('--task', type=str, default="multi_augment_image_folder",
                    help="""task to work on  (default: multi_augment_image_folder).""")
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 4096)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='minimum number of epochs to train (default: 3000)')
parser.add_argument('--download', type=int, default=1,
                    help='download simple datasets like MNIST/CIFAR10 (default: 1)')
parser.add_argument('--image-size-override', type=int, default=224,
                    help='Override and force resizing of images to this specific size (default: None)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--uid', type=str, default="",
                    help='uid for current session (default: empty-str)')


# Model related
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--nce-size', type=int, default=128,
                    help='size for NCE loss computation [output of head] (default: 128)')
parser.add_argument('--nce-temperature', type=float, default=0.1,
                    help='temp for NCE loss (default: 0.1)')
parser.add_argument('--weight-initialization', type=str, default=None,
                    help='weight initialization type; None uses default pytorch init. (default: None)')
parser.add_argument('--model-dir', type=str, default='.models',
                    help='directory which contains saved models (default: .models)')

# Regularizer
parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')
parser.add_argument('--polyak-ema', type=float, default=0, help='Polyak weight averaging co-ef (default: 0)')
parser.add_argument('--convert-to-sync-bn', action='store_true', default=False,#True,
                    help='converts all BNs to SyncBNs (default: True)')

# Optimization related
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping value (default: 0)')
parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                    help='learning rate (default: 0.3)')
parser.add_argument('--lr-update-schedule', type=str, default='cosine',
                    help='learning rate schedule fixed/step/cosine (default: cosine)')
parser.add_argument('--warmup', type=int, default=10, help='warmup epochs (default: 10)')
parser.add_argument('--optimizer', type=str, default="lars_momentum",
                    help="specify optimizer (default: lars_momentum)")
parser.add_argument('--early-stop', action='store_true', default=False,
                    help='enable early stopping (default: False)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs, needs http://url (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

# Device /debug stuff
parser.add_argument('--num-replicas', type=int, default=8,
                    help='number of compute devices available; 1 means just local (default: 8)')
parser.add_argument('--workers-per-replica', type=int, default=2,
                    help='threads per replica for the data loader (default: 2)')
parser.add_argument('--distributed-master', type=str, default='127.0.0.1',
                    help='hostname or IP to use for distributed master (default: 127.0.0.1)')
parser.add_argument('--distributed-port', type=int, default=29300,
                    help='port to use for distributed framework (default: 29300)')
parser.add_argument('--debug-step', action='store_true', default=False,
                    help='only does one step of the execute_graph function per call instead of all minibatches')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')

args = parser.parse_args()

# import half-precision imports
if args.half:
    from apex.fp16_utils import *
    from apex import amp, optimizers


# add aws job ID to config if it exists
aws_instance_id = utils.get_aws_instance_id()
if aws_instance_id is not None:
    args.instance_id = aws_instance_id


class SimCLR(nn.Module):
    """Simple SIMCLR implementation."""

    def __init__(self, base_network_output_size,
                 nce_logits_output_size,
                 classifier_output_size):
        """SimCLR model.

        :param base_network_output_size: output-size of resnet50 embedding
        :param nce_logits_output_size: output-size to use for NCE loss
        :param classifier_output_size: number of classes in classifier problem
        :returns: SimCLR object
        :rtype: nn.Module

        """
        super(SimCLR, self).__init__()
        self.base_network_output_size = base_network_output_size

        # The base network and the head network used for the self-supervised objective
        model_fn = models.__dict__[args.arch]
        self.base_network = nn.Sequential(
            *list(model_fn(pretrained=False).children())[:-1]  # No dense projection
        )
        self.head = nn.Sequential(
            nn.Linear(base_network_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, nce_logits_output_size),
        )

        # The linear classifer head which we will stop-grad to
        self.linear_classifier = nn.Linear(base_network_output_size, classifier_output_size)

    def forward(self, augmentation1, augmentation2):
        """Returns the NCE logits and the linear predictions."""
        representation1 = self.base_network(augmentation1).view(-1, self.base_network_output_size)
        representation2 = self.base_network(augmentation2).view(-1, self.base_network_output_size)
        logits_for_nce1 = self.head(representation1)
        logits_for_nce2 = self.head(representation2)

        # Stop-gradients to the classifier to not learn a trivially better model.
        linear_preds = self.linear_classifier(representation1.clone().detach())
        return logits_for_nce1, logits_for_nce2, linear_preds


def build_lr_schedule(optimizer, last_epoch=-1):
    """ adds a lr scheduler to the optimizer.

    :param optimizer: nn.Optimizer
    :returns: scheduler
    :rtype: optim.lr_scheduler

    """
    if args.lr_update_schedule == 'fixed':
        sched = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0, last_epoch=last_epoch)
    elif args.lr_update_schedule == 'cosine':
        total_epochs = args.epochs - args.warmup
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, last_epoch=last_epoch)
    else:
        raise NotImplementedError("lr scheduler {} not implemented".format(args.lr_update_schedule))

    # If warmup was requested add it.
    if args.warmup > 0:
        warmup = scheduler.LinearWarmup(optimizer, warmup_steps=args.warmup, last_epoch=last_epoch)
        sched = scheduler.Scheduler(sched, warmup)

    return sched


def build_optimizer(model, last_epoch=-1):
    """ helper to build the optimizer and wrap model

    :param model: the model to wrap
    :returns: optimizer wrapping model provided
    :rtype: nn.Optim

    """
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "momentum": functools.partial(optim.SGD, momentum=0.9),
        "lbfgs": optim.LBFGS,
    }

    # Add weight decay (if > 0) and extract the optimizer string
    params_to_optimize = layers.add_weight_decay(model, args.weight_decay)
    full_opt_name = args.optimizer.lower().strip()
    is_lars = 'lars' in full_opt_name
    if full_opt_name == 'lamb':  # Lazy add this.
        assert args.half, "Need fp16 precision to use Apex FusedLAMB."
        optim_map['lamb'] = optimizers.fused_lamb.FusedLAMB

    opt_name = full_opt_name.split('_')[-1] if is_lars else full_opt_name
    print("using {} optimizer {} lars.".format(opt_name, 'with'if is_lars else 'without'))

    # Build the base optimizer
    lr = args.lr * (args.batch_size / 256) if opt_name not in ["adam", "rmsprop"] else args.lr  # Following SimCLR
    opt = optim_map[opt_name](params_to_optimize, lr=lr)

    # Wrap it with LARS if requested
    if is_lars:
        opt = LARS(opt, eps=0.0)

    # Build the schedule and return
    sched = build_lr_schedule(opt, last_epoch=last_epoch)
    return opt, sched


def build_loader_model_grapher(args):
    """builds a model, a dataloader and a grapher

    :param args: argparse
    :param transform: the dataloader transform
    :returns: a dataloader, a grapher and a model
    :rtype: list

    """
    # Build the required transforms for our dataset, eg below:
    train_transform = [
        transforms.CenterCrop(256),
        transforms.RandomResizedCrop((args.image_size_override, args.image_size_override)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        #transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn(x.shape)*0.02)], p=0.5)
    ]
    test_transform = [transforms.CenterCrop(args.image_size_override)]
    loader_dict = {'train_transform': train_transform,
                   'test_transform': test_transform,
                   **vars(args)}
    loader = get_loader(**loader_dict)

    # set the input tensor shape (ignoring batch dimension) and related dataset sizing
    args.input_shape = loader.input_shape
    args.num_train_samples = loader.num_train_samples // args.num_replicas
    args.num_test_samples = loader.num_test_samples  # Test isn't currently split across devices
    args.num_valid_samples = loader.num_valid_samples // args.num_replicas
    args.steps_per_train_epoch = args.num_train_samples // args.batch_size  # drop-remainder
    args.total_train_steps = args.epochs * args.steps_per_train_epoch

    # build the network
    network = SimCLR(base_network_output_size=2048,
                     nce_logits_output_size=args.nce_size,
                     classifier_output_size=loader.output_size)
    network = nn.SyncBatchNorm.convert_sync_batchnorm(network) if args.convert_to_sync_bn else network
    network = network.cuda() if args.cuda else network
    lazy_generate_modules(network, loader.train_loader)
    network = layers.init_weights(network, init=args.weight_initialization)

    if args.num_replicas > 1:
        print("data-paralleling...")
        network = layers.DistributedDataParallelPassthrough(network,
                                                            device_ids=[0],   # set w/cuda environ var
                                                            output_device=0,  # set w/cuda environ var
                                                            find_unused_parameters=True)

    # Get some info about the structure and number of params.
    print(network)
    print("model has {} million parameters.".format(
        utils.number_of_parameters(network) / 1e6
    ))

    # build the grapher object
    grapher = None
    if args.visdom_url and args.gpu == 0:
        grapher = Grapher('visdom', env=utils.get_name(args),
                          server=args.visdom_url,
                          port=args.visdom_port)
    elif args.gpu == 0:
        grapher = Grapher('tensorboard', comment=utils.get_name(args))

    return loader, network, grapher


def lazy_generate_modules(model, loader):
    """ A helper to build the modules that are lazily compiled

    :param model: the nn.Module
    :param loader: the dataloader
    :returns: None
    :rtype: None

    """
    model.eval()
    for augmentation1, augmentation2, labels in loader:
        with torch.no_grad():
            augmentation1 = augmentation1.cuda(non_blocking=True) if args.cuda else augmentation1
            augmentation2 = augmentation2.cuda(non_blocking=True) if args.cuda else augmentation2
            _ = model(augmentation1, augmentation2)
            break

    # initialize the polyak-ema op if it exists
    if hasattr(model, 'polyak_ema') and args.polyak_ema > 0:
        layers.polyak_ema_parameters(model, args.polyak_ema)


def register_plots(loss, grapher, epoch, prefix='train'):
    """ Registers line plots with grapher.

    :param loss: the dict containing '*_mean' or '*_scalar' values
    :param grapher: the grapher object
    :param epoch: the current epoch
    :param prefix: prefix to append to the plot
    :returns: None
    :rtype: None

    """
    if args.gpu == 0 and grapher is not None:  # Only send stuff to visdom once.
        for k, v in loss.items():
            if isinstance(v, dict):
                register_plots(loss[k], grapher, epoch, prefix=prefix)

            if 'mean' in k or 'scalar' in k:
                key_name = '-'.join(k.split('_')[0:-1])
                value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
                grapher.add_scalar('{}_{}'.format(prefix, key_name), value, epoch)


def register_images(output_map, grapher, prefix='train'):
    """ Registers image with grapher. Overwrites the existing image due to space.

    :param output_map: the dict containing '*_img' of '*_imgs' as keys
    :param grapher: the grapher object
    :param prefix: prefix to attach to images
    :returns: None
    :rtype: None

    """
    if args.gpu == 0 and grapher is not None:  # Only send stuff to visdom once.
        for k, v in output_map.items():
            if isinstance(v, dict):
                register_images(output_map[k], grapher, prefix=prefix)

            if 'img' in k or 'imgs' in k:
                key_name = '-'.join(k.split('_')[0:-1])
                img = torchvision.utils.make_grid(v, normalize=True, scale_each=True)
                grapher.add_image('{}_{}'.format(prefix, key_name),
                                  img.detach(),
                                  global_step=0)  # dont use step


def _add_loss_map(loss_tm1, loss_t):
    """ Adds the current dict _t to the previous running dict _tm1

    :param loss_tm1: a dict of previous losses
    :param loss_t: a dict of the current losses
    :returns: a new dict with added values and updated count
    :rtype: dict

    """
    if not loss_tm1:  # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                if not isinstance(v, (float, int, np.float32, np.float64)):
                    resultant[k] = v.detach()
                else:
                    resultant[k] = v

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            if not isinstance(v, (float, np.float32, np.float64)):
                resultant[k] = loss_tm1[k] + v.detach()
            else:
                resultant[k] = loss_tm1[k] + v

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    """ Simply scales all values in the dict by the count

    :param loss_map: the dict of scalars
    :returns: mean of the dict
    :rtype: dict

    """
    for k in loss_map.keys():
        if k == 'count':
            continue

        loss_map[k] /= loss_map['count']

    return loss_map


def execute_graph(epoch, model, loader, grapher, optimizer=None, prefix='test'):
    """ execute the graph; when 'train' is in the name the model runs the optimizer

    :param epoch: the current epoch number
    :param model: the torch model
    :param loader: the train or **TEST** loader
    :param grapher: the graph writing helper (eg: visdom / tf wrapper)
    :param optimizer: the optimizer
    :param prefix: 'train', 'test' or 'valid'
    :returns: dictionary with scalars
    :rtype: dict

    """
    start_time = time.time()
    is_eval = prefix != 'train'
    model.eval() if is_eval else model.train()
    assert optimizer is None if is_eval else optimizer is not None
    loss_map, num_samples = {}, 0

    # iterate over data and labels
    for augmentation1, augmentation2, labels in loader:
        augmentation1 = augmentation1.cuda(non_blocking=True) if args.cuda else augmentation1
        augmentation2 = augmentation2.cuda(non_blocking=True) if args.cuda else augmentation2
        labels = labels.cuda(non_blocking=True) if args.cuda else labels

        with torch.no_grad() if prefix == 'test' else utils.dummy_context():
            if is_eval and args.polyak_ema > 0:                                  # use the Polyak model for predictions
                nce_logits1, nce_logits2, linear_preds = layers.get_polyak_prediction(
                    model, pred_fn=functools.partial(model, augmentation1, augmentation2))
            else:
                nce_logits1, nce_logits2, linear_preds = model(augmentation1,
                                                               augmentation2)    # get normal predictions

            # The loss is the NCE loss + classifer loss (with stop-grad of course).
            acc1, acc5 = metrics.topk(output=linear_preds, target=labels, topk=(1, 5))
            nce_loss = nt_xent(nce_logits1, nce_logits2,
                               temperature=args.nce_temperature,
                               num_replicas=args.num_replicas)
            classifier_loss = F.cross_entropy(input=linear_preds, target=labels)
            loss_t = {
                'loss_mean': nce_loss + classifier_loss,
                'nce_loss_mean': nce_loss,
                'linear_loss_mean': classifier_loss,
                'top1_mean': acc1,
                'top5_mean': acc5,
            }
            loss_map = _add_loss_map(loss_map, loss_t)                           # aggregate loss
            num_samples += augmentation1.size(0)                                 # count minibatch samples

        if not is_eval:                                                          # compute bp and optimize
            optimizer.zero_grad()                                                # zero gradients on optimizer
            if args.half:
                with amp.scale_loss(loss_t['loss_mean'], optimizer) as scaled_loss:
                    scaled_loss.backward()                                       # compute grads (fp16+fp32)
            else:
                loss_t['loss_mean'].backward()                                   # compute grads (fp32)

            if args.clip > 0:
                # TODO: clip by value or norm? torch.nn.utils.clip_grad_value_
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) \
                nn.utils.clip_grad_value_(model.parameters(), args.clip) \
                    if not args.half else optimizer.clip_master_grads(args.clip)

            optimizer.step()
            if args.polyak_ema > 0:                                            # update Polyak mean if requested
                layers.polyak_ema_parameters(model, args.polyak_ema)

            del loss_t

        if args.debug_step:  # for testing purposes
            break

    # compute the mean of the map
    loss_map = _mean_map(loss_map)                                             # reduce the map to get actual means

    # log some stuff
    to_log = '{}-{}[Epoch {}][{} samples][{:.2f} sec]:\t Loss: {:.4f}\tTop-1: {:.4f}\tTop-5: {:.4f}'
    print(to_log.format(
        prefix, args.gpu, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        loss_map['top1_mean'].item(),
        loss_map['top5_mean'].item()))

    # plot the test accuracy, loss and images
    register_plots({**loss_map}, grapher, epoch=epoch, prefix=prefix)

    # tack on images to grapher
    image_map = {'augmentation1_imgs': augmentation1, 'augmentation2_imgs': augmentation2}
    register_images({**image_map}, grapher, prefix=prefix)
    if grapher is not None:
        grapher.save()

    # cleanups (see https://tinyurl.com/ycjre67m) + return loss for early stopping
    loss_val = loss_map['loss_mean'].detach().item()
    loss_map.clear()
    return loss_val


def train(epoch, model, optimizer, train_loader, grapher, prefix='train'):
    """ Helper to run execute-graph for the train dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the train data-loader
    :param grapher: the grapher object
    :param prefix: the default prefix; useful if we have multiple training types
    :returns: mean ELBO scalar
    :rtype: float32

    """
    return execute_graph(epoch, model, train_loader, grapher, optimizer, prefix='train')


def test(epoch, model, test_loader, grapher, prefix='test'):
    """ Helper to run execute-graph for the test dataset

    :param epoch: the current epoch
    :param model: the model
    :param test_loader: the test data-loaderpp
    :param grapher: the grapher object
    :param prefix: the default prefix; useful if we have multiple test types
    :returns: mean ELBO scalar
    :rtype: float32

    """
    return execute_graph(epoch, model, test_loader, grapher, prefix='test')


def init_multiprocessing_and_cuda(rank, num_replicas):
    """Sets the appropriate flags for multi-process jobs."""
    args.gpu = rank  # Set the GPU device to use and correct cuda flags
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)  # Set the cuda device (internal torch fn fails).

    # Set CUDA after setting environment variable.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True
        print("Replica {} / {} using GPU: {}".format(
            rank + 1, num_replicas, torch.cuda.get_device_name(0)))

    # set a fixed seed for GPUs and CPU
    if args.seed is not None:
        print("setting seed %d" % args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)

    if num_replicas > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=args.num_replicas, rank=rank
        )

        # Update batch size appropriately
        args.batch_size = args.batch_size // num_replicas


def run(rank, num_replicas):
    """ Main entry-point into the program

    :param args: argparse
    :returns: None
    :rtype: None

    """
    init_multiprocessing_and_cuda(rank, num_replicas)           # handle multi-process + cuda init logic
    loader, model, grapher = build_loader_model_grapher(args)   # build the model, loader and grapher
    # print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))   # print the config to stdout (after ddp changes)
    optimizer, scheduler = build_optimizer(model)               # the optimizer for the vae
    if args.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # build the early-stopping (or best-saver) objects and restore if we had a previous model
    model = layers.append_save_and_load_fns(model, optimizer, scheduler, grapher, args)
    saver = layers.ModelSaver(model, early_stop=args.early_stop, gpu=args.gpu,
                              burn_in_interval=int(0.1 * args.epochs),  # Avoid tons of saving early on.
                              larger_is_better=False, max_early_stop_steps=10)
    restore_dict = saver.restore()
    init_epoch = restore_dict['epoch']

    # main training loop
    for epoch in range(init_epoch, args.epochs + 1):
        train(epoch, model, optimizer, loader.train_loader, grapher)
        test_loss = test(epoch, model, loader.test_loader, grapher)
        loader.set_all_epochs(epoch)  # set the epoch for distributed-multiprocessing

        # update the learning rate and plot it
        scheduler.step()
        # register_plots({'learning_rate_scalar': scheduler.get_last_lr()[0]}, grapher, epoch)
        register_plots({'learning_rate_scalar': optimizer.param_groups[0]['lr']}, grapher, epoch)

        if saver(test_loss):  # do one more test if we are early stopping
            saver.restore()
            test_loss = test(epoch, model, loader.test_loader, grapher)
            break

        if epoch == 2 and args.gpu == 0:  # make sure we do at least 1 test and train pass
            config_to_post = vars(args)
            slurm_id = utils.get_slurm_id()
            if slurm_id is not None:
                config_to_post['slurm_job_id'] = slurm_id

            grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(config_to_post), 0)

    # cleanups
    if grapher is not None:
        grapher.close()


if __name__ == "__main__":
    print(pprint.PrettyPrinter(indent=4).pformat(vars(args)))
    if args.num_replicas > 1:
        os.environ['MASTER_ADDR'] = args.distributed_master
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        mp.spawn(run, nprocs=args.num_replicas, args=(args.num_replicas,))
    else:
        # Non-distributed launch
        run(rank=0, num_replicas=args.num_replicas)
