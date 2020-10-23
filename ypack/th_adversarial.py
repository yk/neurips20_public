#!/usr/bin/env python3

from ypack import th_utils
import torch as th

def attack_pgd(net_forward_fn, loss_fn_adv, x, y, p='inf', targeted=None, eps=8/255, eps_init=None, lr=1/4, iters=10, random_start=True, debug=False):
    if eps_init is None:
        eps_init = eps
    if debug:
        lr = 1.
    x_orig = x
    x = th.empty_like(x).copy_(x)
    x.requires_grad_(True)
    if random_start:
        x.data += th.empty_like(x).uniform_(-eps_init, eps_init)
    x.data = th.clamp(x.data, -1., 1.)
    for i in range(iters):
        if x.grad is not None:
            x.grad.zero_()

        logits = net_forward_fn(x)
        if targeted is not None:
            if targeted == True:
                loss = th.sum(loss_fn_adv(logits, (y+1)%logits.shape[-1]))
            else:
                loss = th.sum(loss_fn_adv(logits, targeted))
            multiplier = -1
        else:
            loss = th.sum(loss_fn_adv(logits, y))
            multiplier = 1
        loss.backward()

        adv_dir = x.grad * multiplier
        adv_dir = th_utils.normalize(adv_dir, p=p)
        x.data.add_(lr * eps * adv_dir)
        x.data = th_utils.project_to_norm(x.data, x_orig, eps, project_up=False, p=p)
        if debug:
            break
    x = x.detach()
    return x


def attack_deepfool(net_forward_fn, x, y, eps=8/255, eps_init=None, iters=50, overshoot=0.02, numeric_stability_constant=1e-4, binary_search_iters=50, debug=False):
    # adapted from https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
    if eps_init is None:
        eps_init = eps
    x_orig = x
    x = th.empty_like(x).copy_(x)
    x.requires_grad_(True)
    x.data += th.empty_like(x).uniform_(-eps_init, eps_init)
    x.data = th.clamp(x.data, -1., 1.)
    batch_i = th.arange(x.shape[0])
    r_tot = th.zeros_like(x.data)
    for i in range(iters):
        if x.grad is not None:
            x.grad.zero_()

        logits = net_forward_fn(x)
        df_inds = np.argsort(logits.detach().cpu().numpy(), axis=-1)
        df_inds_other, df_inds_orig = df_inds[:, :-1], df_inds[:, -1]
        df_inds_orig = th.from_numpy(df_inds_orig)
        df_inds_orig = df_inds_orig.to(x.device)
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

        r_i = (pert+numeric_stability_constant)[:, None, None, None] * w / th.norm(w.flatten(1), 2, -1)[:, None, None, None]
        r_tot += r_i * not_done_inds[:, None, None, None].float()

        x.data = x_orig + (1. + overshoot) * r_tot
        # x.data = th.clamp(x.data, -1., 1.)
        # r_tot = x.data - x_orig

        if debug:
            break
    x = x.detach()

    dx = x - x_orig
    dx_l_low, dx_l_high = th.zeros_like(dx), th.ones_like(dx)

    #binary search
    for i in range(binary_search_iters):
        dx_l = (dx_l_low + dx_l_high) / 2.
        dx_x = x_orig + dx_l * dx
        dx_y = net_forward(dx_x).argmax(-1)
        label_stay = dx_y == y
        label_change = dx_y != y
        dx_l_low[label_stay] = dx_l[label_stay]
        dx_l_high[label_change] = dx_l[label_change]

    x = dx_x

    return x
