from logging import warning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import warnings
import argparse
import random
import time
import os
import json
import math
from nltk import TreePrettyPrinter, Tree
import matplotlib.pyplot as plt

__has_wandb__ = False
try:
    import wandb
    # don't hardcode the key; get it from ENV VAR of client machine
    # for compute node, the key will be set by XT (from xt_config.yaml)
    # if os.getenv("WANDB_API_KEY"):
    #    __has_wandb__ = True
except BaseException as ex:
    pass

from data import BinaryT2TDataset
from TPR_utils import TPRConverter, TPRManipulator, DecodedTPR2Tree
from models import *
from vector_symbolic_utils import VectorSymbolicConverter


def calculate_loss(decoded):
    # first-order loss (MSE-onehot)
    empty_positions = batch['output'] == 0
    tok_mse_loss = mse_loss(decoded[~empty_positions].view(-1, decoded.size(-1)),
                            F.one_hot(batch['output'][~empty_positions].view(-1),
                                      num_classes=decoded.size(-1)).float())
    # first-order <empty> token loss (mse)
    empty_loss = decoded[empty_positions].view(-1, decoded.size(-1)).norm(p=2, dim=-1)
    total_loss = (tok_mse_loss.sum() + empty_loss.sum()) / (tok_mse_loss.size(0) + empty_loss.size(0))
    # exit earlier if loss is NaN
    if total_loss.isnan():
        exit(0)

    return total_loss


parser = argparse.ArgumentParser(description='pure TPR repo')
# arch
parser.add_argument('--d_filler', type=int, default=None,
                    help='dimension of filler vectors. Set to len(train_vocab) if None.')
parser.add_argument('--d_role', type=int, default=None,
                    help='dimension of role vectors. Set to 2**max_tree_depth if None.')
parser.add_argument('--dtm_steps', type=int, default=3,
                    help='Number of steps for dtm to run.')
parser.add_argument('--op_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                    help='The operation distribution function.')
parser.add_argument('--arg_dist_fn', type=str, default='softmax', choices=['softmax', 'gumbel'],
                    help='The argument distribution function.')
parser.add_argument('--transformer_norm_first', type=int, default=1)
parser.add_argument('--transformer_activation', type=str, default='gelu')
parser.add_argument('--transformer_nheads', type=int, default=4)
parser.add_argument('--router_dropout', type=float, default=0.0,
                    help='Router dropout')
parser.add_argument('--router_hidden_dim', type=int, default=None,
                    help='Router hidden dim')
parser.add_argument('--ctrl_hidden_dim', type=int, default=64,
                    help='Ctrl hidden dim')
parser.add_argument('--ctrl_num_layers', type=int, default=1,)
parser.add_argument('--predefined_ops_random', action='store_true',
                    help='Whether the car/cdr/cons matrices are calculated exactly or learnable random matrices')
# training
parser.add_argument('--epoch', type=int, default=None,
                    help='Number of training epochs. Either this or --steps must be set.')
parser.add_argument('--steps', type=float, default=None,
                    help='Number of training steps. Either this or --epoch must be set.')
parser.add_argument('--batch_size', type=int, default=8,
                    help='random seed')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='./data_files',
                    help='The directory where the data is stored.')
parser.add_argument('--task_type', type=str, default='active_logical_ttb',
                    help='The task type from Basic Sentence Transforms')
parser.add_argument('--use_vsas', action='store_true', help='If specified, use vsas to represent' + 
                    'the binary tree data-structure and use vsa operations to encode car/cdr/cons operations')
parser.add_argument('--vsa_type', default='HRR', choices=['HRR, MBAT', 'FHRR', 'MAP-C', 'MAP-B', 'MAP-I'])
''' 
non self-inverse & approximataely invertible: HRR
non self-inverse & exact invertible: FHRR, MBAT
self-inverse & approximately invertible: MAP-C
self-inverse & exact invertible MAP-B, MAP-I, BSC
Walsh Hadamard Derived VSA (2024 neurips paper )
'''

# optim
parser.add_argument('--optim_beta1', type=float, default=0.9)
parser.add_argument('--optim_beta2', type=float, default=0.98)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'exponential', 'none'])
parser.add_argument('--scheduler_gamma', type=float, default=1, help='The gamma to apply if the scheduler is exponential.')
parser.add_argument('--wd', type=float, default=1e-2)
parser.add_argument('--gclip', type=float, default=1.)
parser.add_argument('--num_warmup_steps', type=float, default=0,
                    help='Number of warmup steps to linearly increase the learning rate from 0 to lr.')

# IO
parser.add_argument('--max_tree_depth', type=int, default=6,
                    help='max depth of input trees')
parser.add_argument('--debug', type=int, default=1, help='1 [debug], 0 [off].')
parser.add_argument('--log_grad_norm', action='store_true',
                    help='Whether to log gradient norms during training.')
parser.add_argument('--save_file', type=str, default='result.tsv')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt')
parser.add_argument('--max_train_examples', type=int, default=None, help='The number of training examples')
parser.add_argument('--validate_every_num_epochs', type=int, default=1,
                    help='How many training epochs do we wait before validating. The default is to validate after'
                         'every training epoch. This is useful when max_train_examples is set to a small number'
                         'so that we don\'t spend too much time validating.')
parser.add_argument('--base_log_dir', type=str, default='logging')
parser.add_argument('--wandb_tag', type=str, default='test')
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--train_log_freq', type=float, default=20., help='training log frequency in steps, -1 logs at the end of each epoch')
parser.add_argument('--do_not_watch_gradients', action='store_true', help='if specified, we do not watch gradients')

args = parser.parse_args()
print(sorted(vars(args).items()))

if args.steps is None and args.epoch is None:
    assert False, 'Either --steps or --epoch must be set.'
elif args.steps is not None and args.epoch is not None:
    assert False, 'Only one of --steps or --epoch can be set.'

if args.router_hidden_dim is None:
    args.router_hidden_dim = args.ctrl_hidden_dim * 4

# LOGGING 
timestamp = time.strftime('%d-%m-%Y-%H:%M:%S')
# wandb stuff
if args.use_wandb:
    wandb.init(project="TPR", dir=args.base_log_dir, config=args.__dict__, tags=[args.wandb_tag],
            mode="online" if args.use_wandb else "disabled", name=args.wandb_name)
    wandb.define_metric("train_acc", summary="max", step_metric='epoch')
    wandb.define_metric("valid_acc", summary="max", step_metric='epoch')
    wandb.define_metric("train_loss", summary="min", step_metric='epoch')
    wandb.define_metric("valid_loss", summary="min", step_metric='epoch')

# TB stuff
tensorboard_dir = os.path.join(args.base_log_dir, 'tb', timestamp)
if not(os.path.exists(tensorboard_dir)):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(log_dir=tensorboard_dir)
writer.add_text('Hyperparameters', '\n'.join(str(x) for x in sorted(vars(args).items())), global_step=0)

# checkpoint stuff 
checkpoint_dir = os.path.join(args.base_log_dir, 'checkpoints')
if not(os.path.exists(checkpoint_dir)): 
    os.makedirs(checkpoint_dir)

# debug
if args.debug:
    debug_dir = os.path.join(args.base_log_dir, 'debug', timestamp)
    if not(os.path.exists(debug_dir)):
        os.makedirs(debug_dir)
    
# general stats
summary_dir = os.path.join(args.base_log_dir, 'summary', timestamp)
if not(os.path.exists(summary_dir)):
        os.makedirs(summary_dir)

# apply seed
if args.seed is None:
    # get a specific seed that we can report with the run
    args.seed = random.randint(0, 65535)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

batch_size = args.batch_size
max_depth = args.max_tree_depth
d_filler = args.d_filler
d_role = 2**max_depth if args.d_role is None else args.d_role

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

task_dir = os.path.join(args.data_dir, args.task_type)

train_data = BinaryT2TDataset(os.path.join(task_dir, 'train.jsonl'), max_depth=max_depth, device=device,
                              max_examples=args.max_train_examples)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print('{} training examples'.format(len(train_data)))

if args.steps is not None:
    args.epoch = math.ceil(args.steps / (len(train_loader)*args.batch_size))
    print('Steps set, training for {} epochs'.format(args.epoch))

valid_data = BinaryT2TDataset(os.path.join(task_dir, 'dev.jsonl'), max_depth=max_depth, ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

test_data = BinaryT2TDataset(os.path.join(task_dir, 'test.jsonl'), max_depth=max_depth, ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


eval_long_loader = None
eval_new_loader = None
if os.path.isfile(os.path.join(task_dir, 'ood_long.jsonl')):
    eval_long_data = BinaryT2TDataset(os.path.join(task_dir, 'ood_long.jsonl'), max_depth=max_depth,
                                      ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind,
                                      device=device)
    eval_long_loader = DataLoader(eval_long_data, batch_size=batch_size, shuffle=False)

if os.path.isfile(os.path.join(task_dir, 'ood_new.jsonl')):
    eval_new_data = BinaryT2TDataset(os.path.join(task_dir, 'ood_new.jsonl'), max_depth=max_depth,
                                     ind2vocab=train_data.ind2vocab, vocab2ind=train_data.vocab2ind, device=device,)
    eval_new_loader = DataLoader(eval_new_data, batch_size=batch_size, shuffle=False)

if d_filler is None:
    d_filler = len(train_data.ind2vocab)

if args.use_vsas:
    raise NotImplementedError('VSA method not yet implemented!!')
else:
    vector_symbolic_converter = TPRConverter(num_fillers=len(train_data.ind2vocab), num_roles=2**max_depth-1,
            d_filler=d_filler, d_role=d_role).to(device=device)
    vector_symbolic_manipulator = TPRManipulator(role_emb=vector_symbolic_converter.role_emb,
                                                 num_ops=3, predefined_ops_random=args.predefined_ops_random)

dtm = DiffTreeMachine(d_filler, d_role, args.ctrl_hidden_dim, vector_symbolic_converter.role_emb, args.dtm_steps,
                      args.router_hidden_dim, nhead=args.transformer_nheads,
                      dropout=args.router_dropout, transformer_activation=args.transformer_activation,
                      transformer_norm_first=args.transformer_norm_first, op_dist_fn=args.op_dist_fn, arg_dist_fn=args.arg_dist_fn,
                      ind2vocab=train_data.ind2vocab, vector_symbolic_converter=vector_symbolic_converter,
                      vector_symbolic_manipulator=vector_symbolic_manipulator).to(device=device)

params = list(dtm.parameters())

trainable_params = list(filter(lambda p: p.requires_grad, params))
print('Trainable params: {}'.format(sum(p.numel() for p in trainable_params)))

optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd, betas=(args.optim_beta1, args.optim_beta2),)

if args.scheduler == 'cosine':
    num_scheduler_epochs = args.epoch - math.ceil(args.num_warmup_steps / len(train_loader)) + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_scheduler_epochs, verbose=False)
elif args.scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)
else:
    scheduler = None


mse_loss = torch.nn.MSELoss(reduction='none')

torch.autograd.set_detect_anomaly(False)
best_train_acc = 0.
best_valid_acc = 0.
step = 0
lr = args.lr

iters = len(train_loader)

if args.use_wandb and not(args.do_not_watch_gradients):
    wandb.watch(dtm, log='gradients', log_freq=1)

# Log on the last training step of each epoch
if args.train_log_freq == -1:
    args.train_log_freq = iters - 1

for epoch_i in range(args.epoch):
    train_correct = 0
    partial_train_correct = 0
    partial_train_total = 0
    accum_i = 0
    train_total = 0
    train_total_loss_accum = 0
    train_started = time.time()

    gumbel_temp = 1

    # Training
    for i, batch in enumerate(train_loader):
        # On the last warmup step, the correct lr will be set
        is_warmup = step < args.num_warmup_steps
        if is_warmup:
            lr = args.lr * step / args.num_warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if (args.op_dist_fn == 'gumbel' or args.arg_dist_fn == 'gumbel') and gumbel_temp > .5:
            gumbel_temp = max(.5, 1 - 1/args.steps * step)
            dtm.set_gumbel_temp(gumbel_temp)
            #print('Gumbel temp:', gumbel_temp)

        bsz = batch['input'].size(0)
        optimizer.zero_grad(set_to_none=True)
        # Use an agent
        output, _, entropies = dtm(vector_symbolic_converter.encode_stree(batch['input']))
        decoded = vector_symbolic_converter.decode_vsymbolic(output, return_similarities=True)

        fully_decoded = DecodedTPR2Tree(decoded)

        # calc/accumulate loss
        total_loss = calculate_loss(decoded)

        train_total_loss_accum += total_loss.item()
        # calc/accumulate train acc
        train_correct += (fully_decoded == batch['output']).all(dim=-1).sum().item()
        train_total += batch['output'].size(0)

        empty_positions = batch['output']==0
        partial_train_correct += (fully_decoded == batch['output'])[~empty_positions].sum().item()
        # The denominator equals the number of non-empty positions plus the number of positions which should
        # be empty but are not
        partial_train_total += (~empty_positions).sum().item() + (fully_decoded != batch['output'])[empty_positions].sum().item()
        accum_i += 1

        total_loss.backward()

        if args.log_grad_norm:
            parameters = dtm.named_parameters()
            parameters = [p for p in parameters if p[1].grad is not None]
            for parameter in parameters:
                writer.add_scalar('grad_norm/{}'.format(parameter[0]), parameter[1].grad.norm(), step)

        # TODO: look into the norm of our gradients, it seems very small
        if args.gclip > 0:
            torch.nn.utils.clip_grad_norm_(dtm.parameters(), args.gclip)

        optimizer.step()

        if step % args.train_log_freq == 0:
            train_total_loss = train_total_loss_accum / accum_i
            writer.add_scalar('Loss/train', train_total_loss, step)
            train_acc = train_correct / train_total
            partial_train_acc = partial_train_correct / partial_train_total
            writer.add_scalar('Accuracy/train', train_acc, step)
            writer.add_scalar('Accuracy/partial_train', partial_train_acc, step)
            accum_i = 0
            train_correct = 0
            train_total = 0
            partial_train_correct = 0
            partial_train_total = 0
            train_total_loss_accum = 0
            if args.use_wandb:
                wandb.log(dict(
                    epoch=epoch_i,
                    train_acc=train_acc,
                    partial_train_acc=partial_train_acc,
                    train_loss=train_total_loss,
                    lr=lr,
                ))

        step += 1

        if args.steps and step > args.steps:
            break

    if epoch_i % args.validate_every_num_epochs == 0 or (args.steps and step == args.steps):
        # Evaluation
        with torch.inference_mode():
            dtm.eval()
            valid_loss_accum = 0
            valid_correct = 0
            valid_total = 0
            partial_valid_correct = 0
            partial_valid_total = 0

            for i, batch in enumerate(valid_loader):
                bsz = batch['input'].size(0)
                is_debug_step = i == 0 if args.debug else False
                output, debug_info, entropies = dtm(vector_symbolic_converter.encode_stree(batch['input']), debug=is_debug_step)

                if is_debug_step:
                    formatted_tree = TreePrettyPrinter(Tree.fromstring(BatchSymbols2NodeTree(batch['output'], dtm.ind2vocab)[0].str()))
                    debug_text = debug_info['text']
                    with open(os.path.join(debug_dir, 'logs.txt'), 'a+') as f: 
                        f.write('\n'.join(debug_text))
                        f.write('Correct output:\n{}'.format(formatted_tree.text()))
                        
                    writer.add_text('Epoch {}'.format(epoch_i), '\n\n'.join(debug_text), global_step=step)
                    
                decoded = vector_symbolic_converter.decode_vsymbolic(output, return_similarities=True)

                fully_decoded = DecodedTPR2Tree(decoded)

                total_valid_loss = calculate_loss(decoded)

                valid_loss_accum += total_valid_loss.item()
                valid_correct += (fully_decoded==batch['output']).all(dim=-1).sum().item()
                valid_total += batch['output'].size(0)

                empty_positions = batch['output'] == 0
                partial_valid_correct += (fully_decoded == batch['output'])[~empty_positions].sum().item()
                # The denominator equals the number of non-empty positions plus the number of positions which should
                # be empty but are not
                partial_valid_total += (~empty_positions).sum().item() + (fully_decoded != batch['output'])[empty_positions].sum().item()

            dtm.train()

            valid_loss = valid_loss_accum / len(valid_loader)

            valid_acc = valid_correct / valid_total
            partial_valid_acc = partial_valid_correct / partial_valid_total
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(dtm.state_dict(), os.path.join(checkpoint_dir, 
                                                          f'{timestamp}-{args.checkpoint_file}'))

            if args.debug:
                # log info to console at end of each epoch
                train_elapsed = time.time() - train_started
                train_rate = len(train_loader) / train_elapsed
                print(f'epoch: {epoch_i:,}')

                print(
                    f'  Train Acc: {train_acc:.2f}, partial_train_acc: {partial_train_acc:.2f}, total_loss: {train_total_loss:.5f}, lr: {lr:.10f}, samples/sec: {train_rate:.0f}, ')
                print(
                    f'  Valid Acc: {valid_acc:.2f}, partial_valid_acc: {partial_valid_acc:.2f}, total_loss: {valid_loss:.5f}')

            # log TRAIN/VALID metrics to Tensorboard
            writer.add_scalar('LR', lr, step)
            writer.add_scalar('Accuracy/valid', valid_acc, step)
            writer.add_scalar('Accuracy/partial valid', partial_valid_acc, step)
            writer.add_scalar('Loss/valid', valid_loss, step)
            if args.use_wandb:
                wandb.log(dict(
                    epoch=epoch_i,
                    valid_acc=valid_acc,
                    valid_loss=valid_loss,
                    partial_valid_acc=partial_valid_acc,
                    lr=lr
                ))

    best_train_acc = max(train_acc, best_train_acc)

    if scheduler and not is_warmup:
        # adjust LR as per scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

def calculate_accuracy(data_loader, data_name):
    with torch.inference_mode():
        correct = 0
        total = 0
        for i, batch in enumerate(data_loader):
            bsz = batch['input'].size(0)
            output, debug_info, _ = dtm(vector_symbolic_converter.encode_stree(batch['input']))

            fully_decoded = DecodedTPR2Tree(vector_symbolic_converter.decode_vsymbolic(output, return_similiarities=True))

            correct += (fully_decoded == batch['output']).all(dim=-1).sum().item()
            total += batch['output'].size(0)
            
        acc = correct/total 
        print(f'{data_name} Acc: {correct / total:.2f}')
        writer.add_scalar('Accuracy/{}'.format(data_name), acc, step)
        
        if args.use_wandb: 
            wandb.log({f'Final accuracy/{data_name}': acc, 'step': step})
        return acc

# Reload the best checkpoint
dtm.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{timestamp}-{args.checkpoint_file}')))

test_acc = calculate_accuracy(test_loader, 'Test')
if eval_long_loader:
    ood_long_acc = calculate_accuracy(eval_long_loader, 'ood_long')

if eval_new_loader:
    ood_new_acc = calculate_accuracy(eval_new_loader, 'ood_new')

with open(args.save_file, 'a') as f:
    log_obj = {'last_train_acc': train_acc,
               'last_valid_acc': valid_correct/valid_total,
               'best_train_acc': best_train_acc,
               'best_valid_acc': best_valid_acc}
    log_obj.update(vars(args))
    f.write(json.dumps(log_obj)+'\n')


print("final metrics & hparams:\n{}\n".format(log_obj))
print(f'Best Train Acc: {best_train_acc:.3f}')
print(f'Best Valid Acc: {best_valid_acc:.3f}')

with open(os.path.join(summary_dir, 'summary.txt'), 'a+') as f: 
    f.write(f'Best train acc: {best_train_acc:.3f}\n')
    f.write(f'Best val acc: {best_valid_acc:.3f}\n')
    f.write(f'Test acc: {test_acc:.3f}\n')
    if eval_long_loader:
        f.write(f'Ood_long test acc: {test_acc:.3f}\n')
    if eval_new_loader:
        f.write(f'Ood_new test acc: {test_acc:.3f}\n')

writer.close()


# change file stuff 
# let's visualise the distribution of weights -> wandb 
# apply our hyperparam search -> try and find optimal hyperparams that lead to best performance on val set with smallest number of epochs
# still needs about 50 epochs to ifner correct rules... (potentially because we are treating the tree like a 'flat' structure...)