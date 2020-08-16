#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division, print_function
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup, edge_aggregator_lookup, \
    metapath_aggregator_lookup
from helpers import set_seeds, to_numpy
from problem import NodeProblem
from models import CLING, MyDataParallel
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import os
from functools import partial
import sys
import argparse
import ujson as json
import numpy as np
from time import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# --
# Helpers

def train_step(model, optimizer, ids, targets, loss_fn):
    optimizer.zero_grad()
    preds, weights = model(ids, train=True)
    if weights is not None:
        weights = weights.cpu().detach().numpy()
        if len(weights.shape) > 1 and weights.shape[0] != 1:
            weights = np.sum(weights, axis=0)/weights.shape[0]
        # print(weights)
    loss = loss_fn(preds, targets.squeeze())
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return loss, preds


def evaluate(model, problem, batch_size, loss_fn, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    loss = 0
    count = 0
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
        # print(ids.shape,targets.shape)
        pred, _ = model(ids, train=False)
        loss += loss_fn(pred, targets.squeeze()).item() * ids.shape[0]
        count += ids.shape[0]
        preds.append(to_numpy(pred))
        acts.append(to_numpy(targets))
    #
    return loss/count, problem.metric_fn(np.vstack(acts), np.vstack(preds))


# def evaluate(model, problem, batch_size, mode='val'):
#     assert mode in ['test', 'val']
#     preds, acts = [], []
#     for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
#         # print(ids.shape,targets.shape)
#         preds.append(to_numpy(model(ids, train=False)))
#         acts.append(to_numpy(targets))
#
#     return problem.metric_fn(np.vstack(acts), np.vstack(preds))
# # --
# Args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem-path', type=str,
                        default='../../../LineGraphGCN/data/dblp2/')
    parser.add_argument('--problem', type=str, default='dblp')
    parser.add_argument('--no-cuda', action="store_true", default=False)

    # Optimization params
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-init', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--lr-patience', type=int, default=10)

    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input-dropout', type=float, default=0)
    parser.add_argument('--batchnorm', action="store_true")
    parser.add_argument('--tolerance', type=int, default=10)
    parser.add_argument('--attn-dropout', type=float, default=0.3)
    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='conch_sampler')
    parser.add_argument('--aggregator-class', type=str, default='attention2')
    parser.add_argument('--prep-class', type=str, default='linear')  # identity
    parser.add_argument('--mpaggr-class', type=str, default='gate')
    parser.add_argument('--edgeupt-class', type=str, default='identity')
    parser.add_argument('--concat-node', action="store_true")
    parser.add_argument('--concat-edge', action="store_true")

    parser.add_argument('--in-edge-len', type=int, default=130)
    parser.add_argument('--in-node-len', type=int, default=128)
    parser.add_argument('--n-hid', type=int, default=512)
    parser.add_argument('--prep-len', type=int, default=256)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-layer', type=int, default=1)
    parser.add_argument('--K', type=int, default=4044)
    parser.add_argument('--train-per', type=float, default=0.4)
    parser.add_argument('--n-train-samples', type=str, default='1000,100')
    parser.add_argument('--n-val-samples', type=str, default='1000,600')
    parser.add_argument('--output-dims', type=str, default='128,128')

    

    # Logging
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--show-test', action="store_true")
    parser.add_argument('--profile', action="store_true")

    # --
    # Validate args

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(
    ), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: aggregator_class not in %s' % str(
        aggregator_lookup.keys())
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


def main(args):

    # Load problem
    mp_index = {'dblp': ['APA', 'APAPA', 'APCPA'],
                'yelp': ['BRURB', 'BRKRB'],
                'yago': ['MAM', 'MDM', 'MWM'],
                'dblp2': ['APA', 'APAPA', 'APCPA'],
                }
    schemes = mp_index[args.problem]
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    problem = NodeProblem(problem_path=args.problem_path,
                          problem=args.problem, device=device, schemes=schemes,
                          train_per=args.train_per,
                          K=args.K, input_edge_dims=args.in_edge_len, emb_len=args.in_node_len)

    # --
    # Define model

    n_train_samples = list(map(int, args.n_train_samples.split(',')))
    n_val_samples = list(map(int, args.n_val_samples.split(',')))
    output_dims = list(map(int, args.output_dims.split(',')))
    model = CLING(**{
        "problem": problem,
        "n_mp": len(schemes),
        "sampler_class": sampler_lookup[args.sampler_class],

        "prep_class": prep_lookup[args.prep_class],
        "prep_len": args.prep_len,
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "mpaggr_class": metapath_aggregator_lookup[args.mpaggr_class],
        "edgeupt_class": edge_aggregator_lookup[args.edgeupt_class],
        "n_head": args.n_head,
        "layer_specs": [
            {
                "n_train_samples": n_train_samples[0],
                "n_val_samples": n_val_samples[0],
                "output_dim": output_dims[0],
                "activation": F.relu,
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
                'n_hid': args.n_hid,
            },
            {
                "n_train_samples": n_train_samples[1],
                "n_val_samples": n_val_samples[1],
                "output_dim": output_dims[1],
                "activation": F.relu,  # lambda x: x
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
                'n_hid': args.n_hid,
            },

            # {
            #     "n_train_samples": n_train_samples[2],
            #     "n_val_samples": n_val_samples[2],
            #     "output_dim": output_dims[2],
            #     "activation": lambda x: x,  # lambda x: x
            #     "concat_node": args.concat_node,
            #     "concat_edge": args.concat_edge,
            # },
        ][:args.n_layer],
        #
        # "lr_init" : args.lr_init,
        # "lr_schedule" : args.lr_schedule,
        # "weight_decay" : args.weight_decay,
        "dropout": args.dropout,
        "input_dropout": args.input_dropout,
        "batchnorm": args.batchnorm,
        "attn_dropout": args.attn_dropout,
        "concat_node": True,
    })

    if args.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # --
    # Define optimizer
    lr = args.lr_init
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.weight_decay, amsgrad=False)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'asgd':
        optimizer = torch.optim.ASGD(
            model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer =='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0.9, centered=False)

    if args.lr_schedule == 'cosinew':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_patience, T_mult=2, eta_min=1e-5, last_epoch=-1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_patience, eta_min=1e-5, last_epoch=-1)
    elif args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor, patience=args.lr_patience, verbose=False, threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    elif args.lr_schedule == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=args.epochs*26, pct_start=0.3,
            anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay,momentum=0.9)
    # print(model, file=sys.stdout)

    # --
    # Train

    set_seeds(args.seed)

    start_time = time()
    val_metric = None
    tolerance = 0
    best_val_loss = 100000
    best_val_acc = 0
    best_result = None
    best_model = None

    for epoch in range(args.epochs):
        # early stopping
        if tolerance > args.tolerance:
            break
        train_loss = 0

        # Train
        _ = model.train()
        id_count = 0
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):

            loss, preds = train_step(
                model=model,
                optimizer=optimizer,
                ids=ids,
                targets=targets,
                loss_fn=problem.loss_fn,
            )
            train_loss += loss.item() * ids.shape[0]
            id_count += ids.shape[0]
            # train_metric = problem.metric_fn(
            #     to_numpy(targets), to_numpy(preds))
            # print(json.dumps({
            #    "epoch": epoch,
            #    "epoch_progress": epoch_progress,
            #    "train_metric": train_metric,
            #    "time": time() - start_time,
            # }, double_precision=5))
            # sys.stdout.flush()

            if args.lr_schedule == 'onecycle':
                scheduler.step()
            if args.lr_schedule in ['cosine', 'cosinew']:
                scheduler.step(epoch + epoch_progress)

        print(json.dumps({
            "epoch": epoch,
            'lr': [optimizer.param_groups[0]['lr']],
            "time": time() - start_time,
            "train_loss": train_loss/id_count,
        }, double_precision=5))
        sys.stdout.flush()

        # Evaluate
        if epoch >= -1:
            _ = model.eval()
            val_loss, val_metric = evaluate(
                model, problem, batch_size=args.val_batch_size, mode='val', loss_fn=problem.loss_fn,)
            # _, test_metric = evaluate(
            #     model, problem, batch_size=8, mode='test', loss_fn=problem.loss_fn,)
            if val_metric['accuracy'] > best_val_acc or (val_metric['accuracy'] == best_val_acc and val_loss < best_val_loss):
                tolerance = 0
                best_val_loss = val_loss
                best_val_acc = val_metric['accuracy']
                best_result = json.dumps({
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metric": val_metric,
                    # "test_metric": test_metric,
                }, double_precision=5)
                best_model = model
            else:
                tolerance += 1

            print(json.dumps({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metric": val_metric,
                # "test_metric": test_metric,
                "tolerance:": tolerance,
            }, double_precision=5))
            sys.stdout.flush()

            if args.lr_schedule == 'plateau':
                scheduler.step(val_loss)

            if args.lr_schedule in ['step']:
                scheduler.step()

    print('-- done --')
    _, test_metric = evaluate(
        best_model, problem, batch_size=args.val_batch_size, mode='test', loss_fn=problem.loss_fn,)
    print(json.dumps({
        # "epoch": epoch,
        # "val_loss": loss,
        # "val_metric": val_metric,
        "test_metric": test_metric,
        # "tolerance:": tolerance,
    }, double_precision=5), file=sys.stderr)
    # print(best_result, file=sys.stderr)
    sys.stdout.flush()

    # if args.show_test:
    #     _ = model.eval()
    #     print(json.dumps({
    #         "test_metric": evaluate(model, problem, batch_size=args.batch_size, mode='test',loss_fn=problem.loss_fn,)
    #     }, double_precision=5))


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    if args.profile:
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # with torch.autograd.profiler.emit_nvtx():
            main(args)
        print(prof.key_averages().table())
    else:
        main(args)
