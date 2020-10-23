#!/usr/bin/env python3

import os
import time
import signal
import threading
import torch as th
import tqdm
import math
from ypack import power
from collections import namedtuple
import numpy as np


NetworkLinearResult = namedtuple('NetworkLinearResult', 'idx x u s v b')


def check_pid():
    while os.getppid() != 1:
        time.sleep(.1)
    os.kill(os.getpid(), signal.SIGKILL)


def init_worker(worker_id):
    thread = threading.Thread(target=check_pid)
    thread.daemon = True
    thread.start()


def gini_coef(a):
    a = th.sort(a, dim=-1)[0]
    n = a.shape[1]
    index = th.arange(1, n+1)[None, :].float()
    return (th.sum((2 * index - n - 1) * a, -1) / (n * th.sum(a, -1)))


def batch_flat(x, from_dim=1):
    new_shape = list(x.shape)[:from_dim] + [-1]
    new_shape = new_shape
    return x.view(*new_shape)


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


def project_to_norm(x, x_orig, eps, p=2, project_up=True):
    dx = x - x_orig
    dx = normalize(dx, eps=eps, p=p, project_up=project_up)
    return x_orig + dx



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


def normalize(x, eps=1., p=2, project_up=True, flatten_dim=1):
    if p == 2:
        x_flat = x.flatten(flatten_dim)
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



def general_iterative_svd(net_forward_fn, x, fb_iters=10, num_vectors=1):

    with th.no_grad():
        x_fb = th.empty_like(x).copy_(x)

        u_list, v_list = [], []

        for vi in range(num_vectors):
            if vi > 0:
                u_base = th.stack(u_list)
                v_base = th.stack(v_list)
            def forward_backward_fn(v):
                with th.no_grad():
                    u = net_forward_fn(v)
                    if vi > 0:
                        u = project_orth(u, u_base)
                    u = normalize(u)
                with th.enable_grad():
                    x_fb_local = x_fb
                    x_fb_local.requires_grad_(True)
                    x_fb_local.retain_grad()
                    if x_fb_local.grad is not None:
                        x_fb_local.grad.zero_()
                    fx = net_forward_fn(x_fb_local)
                    fxu = (fx * u).sum()
                    fxu.backward()
                    v = x_fb_local.grad.detach()
                with th.no_grad():
                    if vi > 0:
                        v = project_orth(v, v_base)
                    v = normalize(v)
                return u, v
            v0 = th.ones_like(x_fb).normal_() / np.sqrt(x_fb.numel() / x_fb.shape[0])
            u1, v1 = power.power_method_asymmetric(forward_backward_fn, v0, lambda u0, v0: False, max_iters=fb_iters, log_max_iters_reached=False)
            u1, v1 = u1.detach(), v1.detach()
            u_list.append(u1)
            v_list.append(v1)
    return u_list, v_list


def network_linear(input_output_fn, x, num_vectors=None, noise_eps=8/255, num_noise_samples=128, on_cpu=False, batch_size=None, with_bias=True):
    """input_output_fn takes x and idx, gives from and to."""
    if batch_size is None:
        batch_size = num_noise_samples
    results = []
    with th.no_grad():
        for idx, x_b in tqdm.tqdm(enumerate(x), total=len(x)):
            x_noisy = x_b.unsqueeze(0) + x_b.new_empty([num_noise_samples, *x_b.shape]).uniform_(-noise_eps, noise_eps)
            outs_from, outs_to = [], []
            for linear_bidx in range(math.ceil(num_noise_samples / batch_size)):
                ofrom, oto = input_output_fn(x_noisy[linear_bidx*batch_size:(linear_bidx+1)*batch_size], idx)
                outs_from.append(ofrom)
                outs_to.append(oto)
            outs_from, outs_to = (batch_flat(th.cat(o, 0)) for o in (outs_from, outs_to))
            w, b = linear_regression(outs_from, outs_to, on_cpu=on_cpu, with_bias=with_bias)
            try:
                u, s, v = th.svd(w.t())
            except RuntimeError:
                if not x.is_cuda:
                    raise
                u, s, v = th.svd(w.t().cpu())
                u, s, v = u.cuda(), s.cuda(), v.cuda()
            if num_vectors:
                u, s, v = u[:, :num_vectors], s[:num_vectors], v[:, :num_vectors]
            res = NetworkLinearResult(idx, x_b.detach().cpu(), u.cpu(), s.cpu(), v.cpu(), b.cpu() if b is not None else None)
            results.append(res)

    u, s, v = zip(*((nlr.u.t(), nlr.s, nlr.v.t()) for nlr in results))
    u, s, v = (th.stack(t).transpose(0, 1) for t in (u, s, v))
    v = v.view([v.shape[0], x.shape[0], *ofrom.shape[1:]])
    u = u.view([v.shape[0], x.shape[0], *oto.shape[1:]])

    return u, s, v
