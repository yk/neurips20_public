#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # noqa

import io
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import torch as th
import torchvision as tv
import torch.functional as F
from torch.autograd import Variable
import math
import tqdm
from filelock import FileLock
import threading
import time
import signal
import numpy as np
import itertools as itt
import scipy.linalg
import scipy.stats
from scipy.spatial.distance import pdist, squareform, correlation
import cifar_model
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
os.system("taskset -p 0xffffffff %d" % os.getpid())  # noqa

from collections import namedtuple

from ypack.batching import StatsAggregator
from ypack import power

import sh
sh.rm('-rf', 'logs')  # noqa

from absl import logging
logging.set_verbosity(logging.INFO)

from tensorboardX.writer import SummaryWriter
swriter = SummaryWriter('logs')

NetworkLinearResult = namedtuple('NetworkLinearResult', 'idx x y u s v b')


def str2bool(x):
    return x.lower() == 'true'

import argparse  # noqa
parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='cifar10')
parser.add_argument('--model', default='cifar10')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--eval_bs', default=256, type=int)
parser.add_argument('--eval_batches', default=None, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--num_evals', default=20, type=int)
parser.add_argument('--train_log_after', default=0, type=int)
parser.add_argument('--stop_after', default=-1, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--optim', default='rmsprop', type=str)
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--targeted', default='pgd', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--attack_lr', default=.25, type=float)
parser.add_argument('--attack_lr_eval', default=None, type=float)
parser.add_argument('--spectral_lr', default=None, type=float)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--eps_rand', default=None, type=float)
parser.add_argument('--eps_attack', default=None, type=float)
parser.add_argument('--eps_attack_eval', default=None, type=float)
parser.add_argument('--eps_noise', default=None, type=float)
parser.add_argument('--eps_spectral', default=0., type=float)
parser.add_argument('--eps_load', default=None, type=float)
parser.add_argument('--rep', default=0, type=int)
parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--num_noise_samples', default=10, type=int)
parser.add_argument('--num_noise_samples_eval', default=None, type=int)
parser.add_argument('--train_data_aug', default=False, type=str2bool)

parser.add_argument('--rowsum_reg', default=0., type=float)
parser.add_argument('--operator_reg', default=0., type=float)
parser.add_argument('--operator_reg_every', default=1, type=int)
parser.add_argument('--entropy_reg', default=0., type=float)
parser.add_argument('--row_reg', default=0., type=float)
parser.add_argument('--second_reg', default=0., type=float)
parser.add_argument('--margin', default=1., type=float)
parser.add_argument('--ujv_reg', default=0., type=float)
parser.add_argument('--ujv_reg_p', default=2, type=int)
parser.add_argument('--fb_iters', default=10, type=int)
parser.add_argument('--fb_iters_eval', default=None, type=int)
parser.add_argument('--linear_bs', default=512, type=int)
parser.add_argument('--full_svds', default=False, type=str2bool)
parser.add_argument('--full_svds_eval', default=False, type=str2bool)
parser.add_argument('--svd_num_vectors', default=0, type=int)
parser.add_argument('--svd_num_vectors_eval', default=None, type=int)
parser.add_argument('--svd_to_layer_eval', default=-1, type=int)
parser.add_argument('--yoshida_reg', default=0., type=float)
parser.add_argument('--yoshida_iters', default=1, type=int)
parser.add_argument('--load_yoshida', default=False, type=str2bool)
parser.add_argument('--load_static', default=False, type=str2bool)
parser.add_argument('--load_dynamic', default=False, type=str2bool)
parser.add_argument('--load_random', default=False, type=str2bool)
parser.add_argument('--load', default=None, type=str)
parser.add_argument('--measure_svd_prop', default=0, type=int)

parser.add_argument('--debug', default=False, type=str2bool)
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--save_outputs', default=False, type=str2bool)
parser.add_argument('--measure_corr', default=False, type=str2bool)
parser.add_argument('--measure_gini', default=False, type=str2bool)
parser.add_argument('--measure_max_diff', default=False, type=str2bool)
parser.add_argument('--measure_expansion', default=False, type=str2bool)
parser.add_argument('--measure_pixel_importance', default=False, type=str2bool)
parser.add_argument('--measure_feature_variance', default=False, type=str2bool)
parser.add_argument('--corr_centered', default=False, type=str2bool)
parser.add_argument('--constrained', default=True, type=str2bool)
parser.add_argument('--clamp_attack', default=False, type=str2bool)
parser.add_argument('--clamp_uniform', default=False, type=str2bool)
parser.add_argument('--train_adv_ratio', default=0., type=float)
parser.add_argument('--train_drp', default=False, type=str2bool)
parser.add_argument('--eval_drp', default=False, type=str2bool)
parser.add_argument('--dropout_lmb', default=1., type=float)
parser.add_argument('--dropout_iters', default=10, type=int)
parser.add_argument('--pca', default=False, type=str2bool)
parser.add_argument('--wdiff_samples', default=32, type=int)
parser.add_argument('--wdiff_stats', default=False, type=str2bool)
parser.add_argument('--maxp_cutoff', default=None, type=float)
parser.add_argument('--pocket_hypothesis_lambda', default=2., type=float)
parser.add_argument('--pocket_hypothesis_lambda_orth', default=None, type=float)
parser.add_argument('--pocket_hypothesis_steps', default=None, type=int)
parser.add_argument('--pocket_hypothesis_sample', default=1, type=int)
parser.add_argument('--measure_nn_dist_ratio', default=False, type=str2bool)
parser.add_argument('--measure_db_dist', default=False, type=str2bool)
parser.add_argument('--df_overshoot', default=0.02, type=float)
parser.add_argument('--df_numeric', default=1e-4, type=float)
parser.add_argument('--df_steps', default=50, type=int)
parser.add_argument('--df_iters', default=50, type=int)
parser.add_argument('--df_search_iters', default=50, type=int)
parser.add_argument('--measure_noise_optimality', default=False, type=str2bool)
parser.add_argument('--pgd_random_start', default=True, type=str2bool)
parser.add_argument('--measure_distance_traveled', default=False, type=str2bool)

args = parser.parse_args()

args.cuda = args.cuda and th.cuda.is_available()

if args.load:
    if args.load == 'clean':
        pass
    elif args.load == 'pgd':
        args.train_adv_ratio = 0.75
    elif args.load == 'yoshida':
        args.load_yoshida = True
    elif args.load == 'static':
        args.load_static = True
    elif args.load == 'static':
        args.load_static = True
    elif args.load == 'dynamic':
        args.load_dynamic = True
    elif args.load == 'random':
        args.load_random = True


args.eps_rand = args.eps_rand or args.eps
args.eps_attack = args.eps_attack or args.eps
eps_attack_scaled = args.eps_attack
if args.attack == 'pgdl2':
    eps_attack_scaled *= np.sqrt(args.img_size * args.img_size * 3)
args.eps_noise = args.eps_noise or args.eps
args.eps_load = args.eps_load or args.eps_attack

args.fb_iters_eval = args.fb_iters_eval or args.fb_iters

args.spectral_lr = args.spectral_lr or args.attack_lr

args.svd_num_vectors_eval = args.svd_num_vectors_eval or args.svd_num_vectors
args.num_noise_samples_eval = args.num_noise_samples_eval or args.num_noise_samples

args.attack_lr_eval = args.attack_lr_eval or args.attack_lr

args.eps_attack_eval = args.eps_attack_eval or args.eps_attack
eps_attack_scaled_eval = args.eps_attack_eval
if args.attack == 'pgdl2':
    eps_attack_scaled_eval *= np.sqrt(args.img_size * args.img_size * 3)


def check_pid():
    while os.getppid() != 1:
        time.sleep(.1)
    os.kill(os.getpid(), signal.SIGKILL)


def init_worker(worker_id):
    thread = threading.Thread(target=check_pid)
    thread.daemon = True
    thread.start()


def SingularValues(kernel, input_shape):
    transform_coefficients = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    svalues = np.linalg.svd(transform_coefficients, compute_uv=False)
    return svalues


def Clip_OperatorNorm(kernel, input_shape, clip_to):
    transform_coefficients = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    U, D, V = np.linalg.svd(transform_coefficients, compute_uv=True, full_matrices=False)
    D_clipped = np.minimum(D, clip_to)
    if kernel.shape[2] > kernel.shape[3]:
        clipped_transform_coefficients = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coefficients = np.matmul(U * D_clipped[..., None, :], V)
    clipped_kernel = np.fft.ifft2(clipped_transform_coefficients, axes=[0, 1]).real
    return clipped_kernel[np.ix_(*[range(d) for d in kernel.shape])]


def W1NormKernel(kernel):
    pass


def gini_coef(a):
    a = th.sort(a, dim=-1)[0]
    n = a.shape[1]
    index = th.arange(1, n+1)[None, :].float()
    return (th.sum((2 * index - n - 1) * a, -1) / (n * th.sum(a, -1)))


def batch_flat(x, from_dim=1):
    new_shape = list(x.shape)[:from_dim] + [-1]
    new_shape = new_shape
    return x.view(*new_shape)

def batch_flat_norm(x, p=2):
    if p == 'inf':
        return batch_flat(th.abs(x)).max(-1)[0]
    return batch_flat(th.abs(x)**p).sum(-1)

def to_channels_last(t):
    return t.permute(0, 2, 3, 1)

def to_channels_first(t):
    return t.permute(0, 3, 1, 2)


def batch_compat(src, trg, inplace=True):
    while len(src.shape) < len(trg.shape):
        if inplace:
            src.unsqueeze_(-1)
        else:
            src = src.unsqueeze(-1)
    return src


def project_orth(x, b):
    xb = x*b
    xb = batch_flat(xb, 2).sum(-1)
    xb = batch_compat(xb, b)
    xbb = xb * b
    x -= xbb.sum(0)
    return x


def linear_regression(X, Y, with_bias=True, on_cpu=False):
    x, y = batch_flat(X), batch_flat(Y)
    if with_bias:
        x = th.cat([x, th.ones_like(x[:, :1])], -1)
    if on_cpu:
        x, y = x.cpu(), y.cpu()
    pix = th.pinverse(x)
    w = th.mm(pix, y)
    if with_bias:
        w, b = w[:-1], w[-1]
        B = b.view(Y.shape[1:])
    else:
        B = None
    W = w.view([*X.shape[1:], *Y.shape[1:]])
    return W, B


def main():
    nrms = dict(imagenet64=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), cifar10=([.5, .5, .5], [.5, .5, .5]))[args.ds]
    transforms = tv.transforms.Compose([
        tv.transforms.Resize((args.img_size, args.img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(*nrms),
    ])

    if args.ds == 'cifar10':
        data_dir = os.path.expanduser('~/data/cifar10')
        os.makedirs(data_dir, exist_ok=True)
        with FileLock(os.path.join(data_dir, 'lock')):
            train_transforms = transforms
            if args.train_data_aug:
                train_transforms = tv.transforms.Compose([
                    tv.transforms.Pad(4),
                    tv.transforms.RandomCrop((32, 32)),
                    tv.transforms.RandomHorizontalFlip(),
                    transforms,
                ])
            train_ds = tv.datasets.CIFAR10(data_dir, train=True, transform=train_transforms, download=True)
            test_ds = tv.datasets.CIFAR10(data_dir, train=False, transform=transforms, download=True)

    train_loader = th.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, worker_init_fn=init_worker)
    test_loader = th.utils.data.DataLoader(test_ds, batch_size=args.eval_bs, shuffle=True, num_workers=1, drop_last=False, pin_memory=True, worker_init_fn=init_worker)

    if args.ds == 'imagenet64':
        if args.model == 'vgg11':
            net = tv.models.vgg11(pretrained=True)
        elif args.model == 'vgg16':
            net = tv.models.vgg16(pretrained=True)
        elif args.model == 'vgg19':
            net = tv.models.vgg19(pretrained=True)
        else:
            raise ValueError('Unknown model: {}'.format(args.model))
    elif args.ds == 'cifar10':
        if args.model == 'tiny':
            net = cifar_model.cifar10_tiny(32, pretrained=args.mode in ('eval', 'tune'), map_location=None if args.cuda else 'cpu')
        elif args.model == 'tinyb':
            net = cifar_model.cifar10_tiny(32, pretrained=args.mode in ('eval', 'tune'), map_location=None if args.cuda else 'cpu', padding=0, trained_adv=args.train_adv_ratio > 0)
        else:
            net = cifar_model.cifar10(
                128, 
                pretrained=args.mode in ('eval', 'tune') and not args.load_random,
                map_location=None if args.cuda else 'cpu', 
                trained_adv_l2_eps=(args.eps_load if (args.train_adv_ratio > 0 and args.mode == 'eval') else 0), 
                trained_yoshida_eps=(args.eps_load if (args.load_yoshida and args.attack == 'pgdl2') else 0),
                trained_static_eps=(args.eps_load if args.load_static else 0),
                trained_dynamic_eps=(args.eps_load if (args.load_dynamic and args.attack == 'pgdl2') else 0),
            )
    print(net)

    def get_layers(with_input=False):
        layers = itt.chain(net.features.children(), net.classifier.children())
        if with_input:
            layers = itt.chain([None], layers)
        return layers


    def get_layer_names(with_input=False):
        names = [l.__class__.__name__ for l in get_layers()]
        if with_input:
            names = ['input'] + names
        return names

    def get_relu_mask(outputs):
        with th.no_grad():
            mask = []
            for o, n in zip(outputs, get_layer_names(with_input=True)):
                if n.lower() == 'relu':
                    m = o > 0
                else:
                    m = th.ones_like(o, dtype=th.uint8)
                m = m.float()
                if args.cuda:
                    m = m.cuda()
                mask.append(m)
        return mask

    if args.cuda:
        net.cuda()

    def net_forward(x, layer_by_layer=False, from_layer=0, dropout=None, pca=None, with_classifier=True, with_relu_mask=None):
        if not layer_by_layer:
            if with_classifier:
                return net(x)
            else:
                return net.features(x)
        outputs = [x]
        for cidx, c in itt.islice(enumerate(net.features.children()), from_layer, None):
            o = outputs[-1]
            if dropout is not None and cidx == 0:
                dprob = dropout[cidx]
                dmask = th.distributions.Bernoulli(probs=dprob).sample()
                o *= dmask
            if with_relu_mask is not None and c.__class__.__name__.lower() == 'relu':
                m = with_relu_mask[cidx+1]
                c = lambda o: o*m
            outputs.append(c(o))
        flat_features = outputs[-1].view(x.size(0), -1)
        if pca:
            flat_features -= pca.th_mean_
            flat_features = th.matmul(flat_features, pca.th_components_T_)
            flat_features[:, 2:5].zero_()
            flat_features = th.matmul(flat_features, pca.th_components_T_.transpose(1, 0))
            flat_features += pca.th_mean_
        if with_classifier:
            outputs.append(net.classifier(flat_features))
        return outputs

    if args.margin > 0.:
        loss_fn = th.nn.MultiMarginLoss(margin=args.margin, reduce=False)
    else:
        loss_fn = th.nn.CrossEntropyLoss(reduce=False)
    loss_fn_adv = th.nn.CrossEntropyLoss(reduce=False)
    if args.cuda:
        loss_fn.cuda()
        loss_fn_adv.cuda()

    def get_outputs(x, y, from_layer=0, dropout=False, pca=None, detach=False, with_relu_mask=None):
        if dropout:
            x.requires_grad_(True)
        outputs = net_forward(x, layer_by_layer=True, from_layer=from_layer, dropout=None, pca=pca, with_relu_mask=with_relu_mask)
        logits = outputs[-1]
        loss = loss_fn(logits, y)
        if dropout:
            grads = th.autograd.grad(loss.sum(), outputs)
            dropout_p = [th.exp(-th.abs(g**2.)*args.dropout_lmb) for g in grads]
            x.requires_grad_(False)
            outputs = net_forward(x, layer_by_layer=True, from_layer=from_layer, dropout=dropout_p, pca=pca)
            logits = outputs[-1]
            loss = loss_fn(logits, y)

        _, preds = th.max(logits, 1)
        if detach:
            outputs = [o.detach().cpu() for o in outputs]
        return outputs, loss, preds

    def attack_deepfool(x, y):
        # adapted from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
        x_orig = x
        x = th.empty_like(x).copy_(x)
        x.requires_grad_(True)
        x.data += th.empty_like(x).uniform_(-args.eps, args.eps)
        x.data = th.clamp(x.data, -1., 1.)
        batch_i = th.arange(x.shape[0])
        r_tot = th.zeros_like(x.data)
        for i in range(args.df_iters):
            if x.grad is not None:
                x.grad.zero_()

            logits = net_forward(x)
            df_inds = np.argsort(logits.detach().cpu().numpy(), axis=-1)
            df_inds_other, df_inds_orig = df_inds[:, :-1], df_inds[:, -1]
            df_inds_orig = th.from_numpy(df_inds_orig)
            if args.cuda:
                df_inds_orig = df_inds_orig.cuda()
            not_done_inds = df_inds_orig == y
            if not_done_inds.sum() == 0:
                logging.info('breaking deepfool after {}'.format(i))
                break

            logits[batch_i, df_inds_orig].sum().backward(retain_graph=True)
            grad_orig = x.grad.data.clone().detach()
            pert = x.data.new_ones(x.shape[0]) * np.inf
            w = th.zeros_like(x.data)

            for inds in df_inds_other.T:
                x.grad.zero_()
                logits[batch_i, inds].sum().backward(retain_graph=True)
                grad_cur = x.grad.data.clone().detach()
                with th.no_grad():
                    w_k = grad_cur - grad_orig
                    f_k = logits[batch_i, inds] - logits[batch_i, df_inds_orig]
                    pert_k = th.abs(f_k) / th.norm(w_k.flatten(1), 2, -1)
                    swi = pert_k < pert
                    if swi.sum() > 0:
                        pert[swi] = pert_k[swi]
                        w[swi] = w_k[swi]

            r_i = (pert+args.df_numeric)[:, None, None, None] * w / th.norm(w.flatten(1), 2, -1)[:, None, None, None]
            r_tot += r_i * not_done_inds[:, None, None, None].float()

            x.data = x_orig + (1. + args.df_overshoot) * r_tot
            # x.data = th.clamp(x.data, -1., 1.)
            # r_tot = x.data - x_orig

            if args.debug:
                break
        x = x.detach()

        dx = x - x_orig
        dx_l_low, dx_l_high = th.zeros_like(dx), th.ones_like(dx)

        #binary search
        for i in range(args.df_search_iters):
            dx_l = (dx_l_low + dx_l_high) / 2.
            dx_x = x_orig + dx_l * dx
            dx_y = net_forward(dx_x).argmax(-1)
            label_stay = dx_y == y
            label_change = dx_y != y
            dx_l_low[label_stay] = dx_l[label_stay]
            dx_l_high[label_change] = dx_l[label_change]

        x = dx_x

        return x

    def normalize(x, eps=1., p=2, project_up=True):
        if p == 2:
            x_flat = x.flatten(1)
            factor = th.norm(x_flat, p=2, dim=1) + 1e-9
            if not project_up:
                factor = th.max(eps * th.ones_like(factor), factor)
            x_flat /= batch_compat(factor, x_flat)
            x_flat *= eps
            return x_flat.view(x.shape)
        elif p == 'inf':
            if project_up:
                x = th.sign(x) * eps
            else:
                x = th.clamp(x, -eps, eps)
            return x
        else:
            raise NotImplementedError()

    def project(x, x_orig, eps, p=2, project_up=True):
        dx = x - x_orig
        if eps == 0:
            dx = th.zeros_like(dx)
        else:
            dx = normalize(dx, eps=eps, p=p, project_up=project_up)
        return x_orig + dx

    def attack_pgd(x, y, eps=eps_attack_scaled, eps_init=args.eps_attack, attack_lr=args.attack_lr, l2=None, relu_mask=None, targeted=None):
        if l2 is None:
            l2 = args.attack == 'pgdl2'
        if targeted is None:
            targeted = args.targeted
        x_orig = x
        x = th.empty_like(x).copy_(x)
        x.requires_grad_(True)
        if args.pgd_random_start:
            x.data += th.empty_like(x).uniform_(-eps_init, eps_init)
        x.data = th.clamp(x.data, -1., 1.)
        for i in range(args.iters):
            if x.grad is not None:
                x.grad.zero_()

            logits = net_forward(x, with_relu_mask=relu_mask)
            if targeted:
                loss = th.sum(loss_fn_adv(logits, (y+1)%logits.shape[-1]))
                multiplier = -1
            else:
                loss = th.sum(loss_fn_adv(logits, y))
                multiplier = 1
            loss.backward()

            if args.constrained:
                if l2:
                    gx = x.grad
                    gx = normalize(gx)
                    if attack_lr >= 0:
                        max_step = min(attack_lr * eps, 2 * eps) # reaches border for sure
                        x.data.add_(max_step * gx * multiplier)
                    else:
                        x.data.copy_(gx * multiplier)
                    x.data = project(x.data, x_orig, eps, project_up=False)
                else:
                    x.data += attack_lr * eps * th.sign(x.grad) * multiplier
                    x.data = th.min(th.max(x.data, x_orig-eps), x_orig+eps)
                x.data = th.clamp(x.data, -1., 1.)
            else:
                x.data += attack_lr * eps * x.grad
            # if args.debug:
                # break
        x = x.detach()
        inf_norm = (x - x_orig).abs().max().cpu().numpy().item()
        if args.clamp_attack:
            with th.no_grad():
                diff = th.sign(x - x_orig) * inf_norm
                x = x_orig + diff
                x = th.clamp(x, -1., 1.)
        # if args.constrained:
            # assert inf_norm < eps * (1.1), 'inf norm {} > {}'.format(inf_norm, eps)
        return x

    eval_after = math.floor(args.epochs * len(train_ds) / args.batch_size / args.num_evals)

    global_step = 0

    def run_train():
        nonlocal global_step  # noqa: E999
        if args.optim == 'rmsprop':
            optim = th.optim.RMSprop(net.parameters(), lr=args.lr)
        elif args.optim == 'adam':
            optim = th.optim.Adam(net.parameters(), lr=args.lr)
        elif args.optim == 'sgd':
            optim = th.optim.SGD(net.parameters(), lr=args.lr)

        yoshida_store = [None for _ in get_layers()]
        logging.info('train')
        for epoch in tqdm.trange(args.epochs):
            for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
                x, y = batch
                if global_step % eval_after == 0:
                    run_eval(True, False, False)

                if args.cuda:
                    x, y = x.cuda(), y.cuda()

                if args.train_adv_ratio > 0:
                    if args.ujv_reg == 0. or batch_idx % 2 == 0:
                        num_adv = round(x.shape[0] * args.train_adv_ratio)
                        x_adv = attack_pgd(x[:num_adv], y[:num_adv])
                        x = th.cat((x, x_adv))
                        y = th.cat((y, y[:num_adv]))

                net.zero_grad()
                outs, loss, _ = get_outputs(x, y, dropout=args.train_drp)
                if args.entropy_reg > 0.:
                    logits = outs[-1]
                    logits = logits[th.ones_like(logits).scatter_(-1, y[:, None], 0.) == 1.].view(logits.shape[0], -1)
                    entropy = th.distributions.Categorical(logits=logits).entropy()
                    loss -= args.entropy_reg * entropy
                if args.row_reg:
                    w = next(net.classifier.children()).weight
                    rowdiff = w[None, :, :] - w[:, None, :]
                    loss += (rowdiff**2.).sum() * args.row_reg
                if args.second_reg:
                    logits = outs[-1]
                    slogits = logits.sort(-1)[0]
                    sdiff = slogits[:, -1] - slogits[:, -2]
                    loss -= th.log1p(sdiff).mean() * args.second_reg

                loss = loss.mean()

                if args.rowsum_reg > 0.:

                    with th.no_grad():
                        x_pert = th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand) + x
                        outs_pert = net_forward(x_pert, layer_by_layer=True)
                        diffs = [th.abs(o-oo).view(o.shape[0], -1).max(-1)[0] for o, oo in zip(outs, outs_pert)]
                        ratios = [(o_out/o_in).max() for o_out, o_in in zip(diffs[1:], diffs[:-1])]

                    for layer, ratio in zip(get_layers(), ratios):
                        if hasattr(layer, 'weight') and layer.__class__.__name__ != 'BatchNorm2d':
                            if len(layer.weight.shape) == 2:
                                W_inf = th.abs(layer.weight).sum(1)
                            elif len(layer.weight.shape) == 4:
                                W_inf = th.abs(layer.weight).sum(3).sum(2).sum(1)
                            loss += args.rowsum_reg * (th.relu(W_inf - ratio)**2.).sum()

                if args.ujv_reg > 0. and (args.train_adv_ratio == 0. or batch_idx % 2 == 1):
                    if args.full_svds:
                        u_list, v_list, _ = network_full_svd(x, y)
                    else:
                        p_norm = args.ujv_reg_p or 'inf'
                        u_list, v_list, _ = network_svd(x, p_norm=p_norm)
                    uJv_list = []
                    relu_mask = get_relu_mask(outs)
                    for u, v in zip(u_list, v_list):
                        if x.is_cuda:
                            u = u.cuda()
                            v = v.cuda()
                        # Jv = net_forward(v, layer_by_layer=True, with_classifier=False, with_relu_mask=relu_mask)[-1]
                        # uJv = batch_flat(u * Jv).sum(-1)
                        # uJv_list.append(uJv)
                        xpv = x + eps_attack_scaled * v
                        outs_xpv = net_forward(xpv, layer_by_layer=True, with_classifier=True)
                        fdiff = batch_flat(outs[-1] - outs_xpv[-1])
                        uJv_list.append((fdiff**2.).sum(-1))
                    for uJv in uJv_list:
                        # loss += args.ujv_reg * (uJv**2.).mean()
                        loss += args.ujv_reg * uJv.mean()

                if args.yoshida_reg > 0:
                    for layer_idx, layer in enumerate(get_layers()):
                        if hasattr(layer, 'weight') and layer.__class__.__name__ in ('Conv2d', 'Linear'):
                            layer_w = layer.weight
                            w = layer_w
                            if len(w.shape) > 2:
                                w = w.view(w.shape[0], -1)
                            with th.no_grad():
                                v = yoshida_store[layer_idx]
                                if v is None:
                                    v = th.ones_like(w[0]).normal_().unsqueeze(1)
                                v /= v.norm() + 1e-9
                                for _ in range(args.yoshida_iters):
                                    u = th.mm(w, v)
                                    u /= u.norm() + 1e-9
                                    v = th.mm(w.t(), u)
                                    v /= v.norm() + 1e-9
                                yoshida_store[layer_idx] = v
                        uwv = th.mm(th.mm(u.t(), w), v)[0, 0]
                        loss += args.yoshida_reg * (uwv**2.)

                net.zero_grad()
                loss.backward()
                optim.step()

                if args.operator_reg > 0. and (global_step % args.operator_reg_every == 0 or global_step + 1 % eval_after == 0):

                    with th.no_grad():
                        x_pert = th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand) + x
                        outs_pert = net_forward(x_pert, layer_by_layer=True)
                        l2s = [th.norm(th.abs(o-oo).view(o.shape[0], -1), 2, -1) for o, oo in zip(outs, outs_pert)]
                        ratios = [(o_out/o_in).max().cpu().numpy().item() for o_out, o_in in zip(l2s[1:], l2s[:-1])]

                    for idx, (layer, ratio) in enumerate(zip(get_layers(), ratios)):
                        if hasattr(layer, 'weight') and layer.__class__.__name__ != 'BatchNorm2d':
                            w_np = layer.weight.detach().cpu().numpy()
                            if len(layer.weight.shape) == 2:
                                U, D, V = np.linalg.svd(w_np, compute_uv=True, full_matrices=False)
                                D_clipped = np.minimum(D, args.operator_reg*ratio)
                                w_np_clip = np.matmul(U, D_clipped[:, None] * V)
                            elif len(layer.weight.shape) == 4:
                                w_np_clip = Clip_OperatorNorm(w_np, (outs[idx+1].shape[1], outs[idx].shape[1]), args.operator_reg*ratio)
                            layer.weight.data.copy_(th.from_numpy(w_np_clip))

                global_step += 1

        with open('logs/model.ckpt', 'wb') as f:
            th.save(net.state_dict(), f)

    def get_noisy_outputs(x, y, relu_mask=None):
        x_noisy = []
        all_outputs, all_loss, all_preds = [], [], []
        for _ in range(args.num_noise_samples):
            x_rand = x + th.empty_like(x).uniform_(-args.eps_noise, args.eps_noise)
            outputs, loss, preds = get_outputs(x_rand, y, with_relu_mask=relu_mask)
            outputs = [o.detach().cpu() for o in outputs]
            loss, preds = loss.detach().cpu(), preds.detach().cpu()
            x_noisy.append(x_rand)
            all_outputs.append(outputs)
            all_loss.append(loss)
            all_preds.append(preds)
        x_noisy = th.stack(x_noisy, 1)
        all_outputs = [th.stack(outs, 1) for outs in zip(*all_outputs)]
        all_loss, all_preds = th.stack(all_loss, 1), th.stack(all_preds, 1)
        return x_noisy, all_outputs, all_loss, all_preds

    def network_linear(x, y, eps=1., with_relu_mask=True, from_layer=0, to_layer=-1, num_vectors=None, num_samples=args.num_noise_samples, on_cpu=False):
        old_train = net.training
        net.train(False)
        results = []
        with th.no_grad():
            if with_relu_mask:
                outputs = net_forward(x, layer_by_layer=True)
                relu_mask = get_relu_mask(outputs)
            for idx, (x_b, y_b) in tqdm.tqdm(enumerate(zip(x, y)), total=len(x)):
                if with_relu_mask:
                    m = [m[idx].unsqueeze(0) for m in relu_mask]
                else:
                    m = None
                x_noisy = x_b.unsqueeze(0) + x_b.new_empty([num_samples, *x_b.shape]).uniform_(-args.eps_attack, args.eps_attack)
                outs_from, outs_to = [], []
                for linear_bidx in range(math.ceil(num_samples / args.linear_bs)):
                    outs_noisy = net_forward(x_noisy[linear_bidx*args.linear_bs:(linear_bidx+1)*args.linear_bs], layer_by_layer=True, with_relu_mask=m)
                    outs_from.append(outs_noisy[from_layer])
                    outs_to.append(outs_noisy[to_layer])
                outs_from, outs_to = (batch_flat(th.cat(o, 0)) for o in (outs_from, outs_to))
                w, b = linear_regression(outs_from, outs_to, on_cpu=on_cpu)
                try:
                    u, s, v = th.svd(w.t())
                except RuntimeError:
                    if not x.is_cuda:
                        raise
                    u, s, v = th.svd(w.t().cpu())
                    u, s, v = u.cuda(), s.cuda(), v.cuda()
                if num_vectors:
                    u, s, v = u[:, :num_vectors], s[:num_vectors], v[:, :num_vectors]
                res = NetworkLinearResult(idx, x_b.detach().cpu(), y_b.detach().cpu(), u.cpu(), s.cpu(), v.cpu(), b.cpu())
                results.append(res)

        u, s, v = zip(*((nlr.u.t(), nlr.s, nlr.v.t()) for nlr in results))
        u, s, v = (th.stack(t).transpose(0, 1) for t in (u, s, v))
        v = v.view([v.shape[0], *outputs[from_layer].shape])
        u = u.view([v.shape[0], *outputs[to_layer].shape])

        net.train(old_train)
        return u, s, v

    def network_full_svd(x, y, with_relu_mask=True, from_layer=0, to_layer=-1, num_vectors=args.svd_num_vectors, num_samples=args.num_noise_samples):
        u, s, v = network_linear(x, y, with_relu_mask=with_relu_mask, from_layer=from_layer, to_layer=to_layer, num_vectors=num_vectors, num_samples=num_samples)
        return u, v, s

    def network_svd(x, fb_iters=args.fb_iters, p_norm=2, num_vectors=args.svd_num_vectors, from_layer=0, to_layer=-1): # XXX
        old_train = net.training
        net.train(False)
        with th.no_grad():
            outputs = net_forward(x, layer_by_layer=True)
        relu_mask = get_relu_mask(outputs)

        x_fb = th.empty_like(outputs[from_layer]).copy_(outputs[from_layer])
        do_dynamic = args.eps_spectral > 0.

        if to_layer is None:
            to_layer = -2
        if to_layer < 0:
            to_layer += 1
        if to_layer > 0:
            to_layer -= from_layer

        u_list, v_list = [], []

        for vi in range(num_vectors):
            if vi > 0:
                u_base = th.stack(u_list)
                v_base = th.stack(v_list)
            def forward_backward_fn(v):
                nonlocal x_fb
                u = net_forward(v, layer_by_layer=True, with_classifier=True, with_relu_mask=relu_mask, from_layer=from_layer)[to_layer].detach()
                if vi > 0:
                    u = project_orth(u, u_base)
                u = normalize(u)
                with th.enable_grad():
                    # if do_dynamic:
                        # x_fb_local = th.empty_like(x_fb).copy_(x_fb)
                    # else:
                        # x_fb_local = x_fb
                    x_fb_local = th.empty_like(x_fb).copy_(x_fb)
                    x_fb_local.requires_grad_(True)
                    x_fb_local.retain_grad()
                    if x_fb_local.grad is not None:
                        x_fb_local.grad.zero_()
                    fx = net_forward(x_fb_local, layer_by_layer=True, from_layer=from_layer, with_classifier=True)[to_layer]
                    fxu = (fx * u).sum()
                    net.zero_grad()
                    fxu.backward()
                    v = x_fb_local.grad.detach()
                if vi > 0:
                    v = project_orth(v, v_base)
                v = normalize(v, p=p_norm)
                # if do_dynamic:
                    # eps = args.eps_spectral * eps_attack_scaled
                    # x_fb += args.spectral_lr * eps * v
                    # x_fb = project(x_fb, x.data, eps, p=p_norm, project_up=False)
                eps = args.eps_spectral * eps_attack_scaled
                x_fb += args.spectral_lr * eps * v
                x_fb = project(x_fb, x.data, eps, p=p_norm, project_up=False)
                return u, v
            # v0 = th.ones_like(x_fb) / np.sqrt(x_fb.numel() / x_fb.shape[0])
            v0 = th.ones_like(x_fb).normal_() / np.sqrt(x_fb.numel() / x_fb.shape[0])
            u1, v1 = power.power_method_asymmetric(forward_backward_fn, v0, lambda u0, v0: args.debug, max_iters=fb_iters)
            u1, v1 = u1.detach(), v1.detach()
            u_list.append(u1)
            v_list.append(v1)
        uJv_list = []
        for u, v in zip(u_list, v_list):
            Jv = net_forward(v, layer_by_layer=True, with_classifier=True, with_relu_mask=relu_mask, from_layer=from_layer)[to_layer]
            uJv = batch_flat(u * Jv).sum(-1)
            uJv_list.append(uJv)
        u_list = [u.detach().cpu() for u in u_list]
        v_list = [v.detach().cpu() for v in v_list]
        net.train(old_train)
        return u_list, v_list, uJv_list

    def run_eval(with_attack=True, save_outputs=args.save_outputs, measure_corr=args.measure_corr):
        logging.info('eval')
        net.train(False)

        stats = StatsAggregator()

        for eval_batch_i, eval_batch in enumerate(tqdm.tqdm(itt.islice(test_loader, args.eval_batches), total=min(args.eval_batches, len(test_loader)) if args.eval_batches else len(test_loader))):
            x, y = eval_batch
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            def _eval_x(x, tag, outputs_clean=None, v_list_clean=None, v_local_list_clean=None):
                with th.no_grad():
                    outputs, loss, preds = get_outputs(x, y, detach=True)
                    stats.add_batch('loss_{}'.format(tag), loss)
                    stats.add_batch('acc_{}'.format(tag), th.eq(preds, y).float())

                    relu_mask_x = get_relu_mask(outputs)

                    if outputs_clean is not None:
                        relu_mask_fixed = get_relu_mask(outputs_clean)
                        outputs_fixed, loss_fixed, preds_fixed = get_outputs(x, y, detach=True, with_relu_mask=relu_mask_fixed)
                        stats.add_batch('loss_fixed_{}'.format(tag), loss_fixed)
                        stats.add_batch('acc_fixed_{}'.format(tag), th.eq(preds_fixed, y).float())
                        shared_masks = []
                        for idx, (n, mx, mf) in enumerate(zip(get_layer_names(True), relu_mask_x, relu_mask_fixed)):
                            if not n.lower() == 'relu':
                                continue
                            shared_mask = batch_flat(mx == mf)
                            stats.add_batch('shared_relu_mask_fixed_{}_{}_{}'.format(tag, idx, n), shared_mask)
                            shared_masks.append(shared_mask)
                        shared_masks = th.cat(shared_masks, 1)
                        stats.add_batch('shared_relu_mask_fixed_{}_all_all'.format(tag), shared_masks)
                        features_diff = th.norm(batch_flat(outputs_fixed[-2] - outputs[-2]), dim=-1)
                        stats.add_batch('feature_dist_fixed_{}'.format(tag), features_diff)

                    # x_noisy, noisy_outputs, _, _ = get_noisy_outputs(x, y)


                    v_local_list = []
                    if args.svd_num_vectors_eval > 0:
                        if args.full_svds_eval:
                            u_list, v_list, uJv_list = network_full_svd(x, y, num_vectors=args.svd_num_vectors_eval, num_samples=args.num_noise_samples_eval, to_layer=args.svd_to_layer_eval)
                            if args.measure_svd_prop > 0:
                                layer_idcs = [idx for idx, ln in enumerate(get_layer_names()) if ln in ('Conv2d',)]
                                v_global = v_list[0]
                                if x.is_cuda:
                                    v_global = v_global.cuda()
                                outs_v1 = [t.cpu() for t in net_forward(v_global, layer_by_layer=True)]
                                for lidx_i, lidx in enumerate(layer_idcs):
                                    u_list_local, s_list_local, v_list_local = network_linear(x, y, from_layer=lidx, to_layer=-1, num_vectors=args.measure_svd_prop, num_samples=args.num_noise_samples_eval, on_cpu=lidx_i < len(layer_idcs) - 3)
                                    out_v1 = outs_v1[lidx]
                                    out_v1 = out_v1.unsqueeze(0)
                                    ov1 = batch_flat(out_v1, 2)
                                    vl = batch_flat(v_list_local, 2)
                                    corr = th.abs((ov1 * vl).sum(-1)) / th.norm(ov1, dim=-1) / th.norm(vl, dim=-1)
                                    for vidx, vcorr in enumerate(corr):
                                        stats.add_batch('piecewise_corr_vglobal_{}_{}_{}_{}'.format(lidx, get_layer_names()[lidx], vidx, tag), vcorr)
                                    v_local_list.append(v_list_local)
                                    if v_local_list_clean:
                                        vl = batch_flat(v_local_list_clean[lidx_i], 2)
                                        out_x = outputs[lidx]
                                        out_clean_x = outputs_clean[lidx]
                                        dx = out_x - out_clean_x
                                        dx = dx.unsqueeze(0)
                                        dx = batch_flat(dx, 2)
                                        corr = th.abs((dx * vl).sum(-1)) / th.norm(dx, dim=-1) / th.norm(vl, dim=-1)
                                        for vidx, vcorr in enumerate(corr):
                                            stats.add_batch('piecewise_corr_x_{}_{}_{}_{}'.format(lidx, get_layer_names()[lidx], vidx, tag), vcorr)

                        else:
                            u_list, v_list, uJv_list = network_svd(x, fb_iters=args.fb_iters_eval, num_vectors=args.svd_num_vectors_eval)
                            if args.measure_svd_prop > 0:
                                layer_idcs = [idx for idx, ln in enumerate(get_layer_names()) if ln in ('Conv2d',)]
                                v_global = v_list[0]
                                if x.is_cuda:
                                    v_global = v_global.cuda()
                                outs_v1 = [t.cpu() for t in net_forward(v_global, layer_by_layer=True)]
                                for lidx in layer_idcs:
                                    u_list_local, v_list_local, s_list_local = network_svd(x, from_layer=lidx, to_layer=lidx+1, num_vectors=args.measure_svd_prop, fb_iters=args.fb_iters_eval)
                                    v_list_local = th.stack(v_list_local, 0)
                                    out_v1 = outs_v1[lidx]
                                    out_v1 = out_v1.unsqueeze(0)
                                    ov1 = batch_flat(out_v1, 2)
                                    vl = batch_flat(v_list_local, 2)
                                    corr = th.abs(ov1 * vl) / th.norm(ov1, dim=-1, keepdim=True) / th.norm(vl, dim=-1, keepdim=True)
                                    for vidx, vcorr in enumerate(corr):
                                        stats.add_batch('piecewise_corr_{}_{}_{}_{}'.format(lidx, get_layer_names()[lidx], vidx, tag), vcorr)
                        u1, v1, uJv = u_list[0], v_list[0], uJv_list[0]

                        for uJv_i, uJv in enumerate(uJv_list):
                            stats.add_batch('uJv_{}_{}'.format(uJv_i, tag), uJv)

                    else:
                        v1 = th.zeros_like(x)
                        v_list = None
                    if x.is_cuda:
                        v1 = v1.cuda()


                    xv1 = x + eps_attack_scaled * v1

                    outputs_xv1, loss_xv1, preds_xv1 = get_outputs(xv1, preds, detach=True)
                    stats.add_batch('loss_xv1_{}'.format(tag), loss_xv1)
                    stats.add_batch('acc_xv1_{}'.format(tag), th.eq(preds_xv1, preds).float())
                    stats.add_batch('true_acc_xv1_{}'.format(tag), th.eq(preds_xv1, y).float())

                    if v_list_clean is not None:
                        assert outputs_clean is not None
                        x_clean = outputs_clean[0]
                        dev_dir = x.detach().cpu() - x_clean
                        for idx, (v_clean, v) in enumerate(zip(v_list_clean, v_list)):
                            dev_aligns = [np.abs(1 - correlation(dx.numpy(), dv.numpy())) for dx, dv in zip(batch_flat(dev_dir), batch_flat(v_clean))]
                            stats.add_batch('dev_align_v_{}_corr_{}'.format(idx, tag), dev_aligns)
                            v_aligns = [np.abs(1 - correlation(v.numpy(), vc.numpy())) for v, vc in zip(batch_flat(v), batch_flat(v_clean))]
                            stats.add_batch('v_align_v_{}_corr_{}'.format(idx, tag), v_aligns)


                    relu_mask_xv1 = get_relu_mask(outputs_xv1)

                    outputs_masked, loss_masked, preds_masked = get_outputs(x, preds, detach=True, with_relu_mask=relu_mask_xv1)
                    stats.add_batch('loss_masked_{}'.format(tag), loss_masked)
                    stats.add_batch('acc_masked_{}'.format(tag), th.eq(preds_masked, preds).float())
                    stats.add_batch('true_acc_masked_{}'.format(tag), th.eq(preds_masked, y).float())


                    shared_masks = []
                    for idx, (n, mx, mv) in enumerate(zip(get_layer_names(True), relu_mask_x, relu_mask_xv1)):
                        if not n.lower() == 'relu':
                            continue
                        shared_mask = batch_flat(mx == mv)
                        stats.add_batch('shared_relu_mask_xv1_{}_{}_{}'.format(tag, idx, n), shared_mask)
                        shared_masks.append(shared_mask)
                    shared_masks = th.cat(shared_masks, 1)
                    stats.add_batch('shared_relu_mask_xv1_{}_all_all'.format(tag), shared_masks)

                return outputs, loss, preds, v_list, v_local_list

            outputs_clean, loss_clean, preds_clean, v_list_clean, v_local_list_clean = _eval_x(x, 'clean')

            if with_attack:
                x_rand = x + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                _eval_x(x_rand, 'rand', outputs_clean=outputs_clean, v_list_clean=v_list_clean, v_local_list_clean=v_local_list_clean)

                x_pgd = attack_pgd(x, preds_clean, eps=eps_attack_scaled_eval, eps_init=args.eps_attack_eval, attack_lr=args.attack_lr_eval)
                outputs_pgd, _, _, v_list_pgd, v_local_list_pgd = _eval_x(x_pgd, 'pgd', outputs_clean=outputs_clean, v_list_clean=v_list_clean, v_local_list_clean=v_local_list_clean)

                # x_pgd_masked = attack_pgd(x, preds_clean, relu_mask=get_relu_mask(outputs_clean))
                # _eval_x(x_pgd_masked, 'pgd_masked', outputs_clean=outputs_clean, v_list_clean=v_list_clean, v_local_list_clean=v_local_list_clean)

                x_pand = x_pgd + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                _eval_x(x_pand, 'pand', outputs_clean=outputs_pgd, v_list_clean=v_list_pgd, v_local_list_clean=v_local_list_pgd)

                x_init = th.empty_like(x).uniform_(th.min(x), th.max(x))
                outputs_init, _, _, v_list_init, v_local_list_init = _eval_x(x_init, 'init', outputs_clean=outputs_clean, v_list_clean=v_list_clean, v_local_list_clean=v_local_list_clean)

                x_rinit = x_init + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                _eval_x(x_rinit, 'rinit', outputs_clean=outputs_init, v_list_clean=v_list_init, v_local_list_clean=v_local_list_init)

            if args.debug:
                break

        for tag in sorted(stats):
            m, s = stats[tag].mean(), stats[tag].std()
            swriter.add_scalar(tag, m, global_step=global_step)
            swriter.add_scalar('{}_std'.format(tag), s, global_step=global_step)
            logging.info('{}:{}: {:.5f} ({:.5f})'.format(global_step, tag, m, s))



        net.train(True)

    if args.mode == 'eval':
        for p in net.parameters():
            p.requires_grad_(False)
        run_eval()
    elif args.mode in ('train', 'tune'):
        run_train()

    swriter.close()

if __name__ == '__main__':
    main()
