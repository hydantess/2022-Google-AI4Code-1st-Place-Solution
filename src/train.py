# coding=utf-8
import numpy as np
import pandas as pd
import os
import random
import time
import gc
import argparse
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, AdamW, get_cosine_schedule_with_warmup
from parameter import Parameter
from utils import *
from models import *
from data_processing import get_data
from dataset import MarkdownDataset, Collate
import warnings
from transformers import BertTokenizer, RobertaTokenizerFast, AutoTokenizer

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--out_dir', default='../user_data', type=str,
                    help='destination where trained network should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for training')
parser.add_argument('--model_name', default='roberta-base', type=str)
parser.add_argument('--base_epoch', default=5, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=8, type=int, help='train mini-batch size')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--n_accumulate', default=1, type=int)
parser.add_argument('--max_grad_norm', default=-1, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--seq_length', default=512, type=int)
parser.add_argument('--max_seq_length', default=2048, type=int)
parser.add_argument('--folds', default='', type=str)
parser.add_argument('--pre_epoch', default=15, type=int)

args = parser.parse_args()
parameter = Parameter()
parameter.set(**args.__dict__)

log_out_dir = os.path.join(parameter.result_dir, 'logs')
os.makedirs(log_out_dir, exist_ok=True)
log_dir = os.path.join(log_out_dir, '{}.txt'.format(args.model_name))
if os.path.exists(log_dir):
    os.remove(log_dir)
log = Logger()
log.open(log_dir, mode='a')

model_dir = os.path.join(parameter.result_dir, 'models')
os.makedirs(model_dir, exist_ok=True)


def main():
    log.write('>> parameter:\n{}\nargs:{}\n'.format(parameter, args))
    # set random seeds
    seed_everything(parameter.random_seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.write('>> data_processing...\n')
    debug = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')
    #     if debug:
    #         return 1
    # load data
    train_df, code_df_valid = get_data()
    oof_prediction = np.zeros((len(train_df)))
    eval_loss = []
    log.write('---------------------------------------------------------------------------------------------------\n')
    for fold in range(parameter.k_folds):
        if len(args.folds) > 0 and str(fold) not in args.folds.split(','):
            continue
        model = MarkdownModel(get_model_path(args.model_name), seq_length=parameter.seq_length, pretrained=True)
        model.zero_grad()
        if parameter.use_cuda:
            model = model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        log.write('model total parameters:{}\n'.format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.write('model total training parameters:{}\n'.format(total_trainable_params))
        tokenizer = AutoTokenizer.from_pretrained(get_model_path(args.model_name))
        train_model_filename = os.path.join(model_dir, args.model_name + '_fold{}.pth.tar'.format(fold))
        if not os.path.exists(train_model_filename):
            # model.load_state_dict(torch.load(train_model_filename, map_location='cpu')['state_dict'])
            filename = os.path.join(model_dir, args.model_name + '_pre_{}.pth.tar'.format(args.pre_epoch))
            if os.path.exists(filename):
                model.encoder.load_state_dict(torch.load(filename)['state_dict'])
                log.write('loaded pre-trained model\n')
            else:
                log.write('not loaded pre-trained model\n')
            collate_fn = Collate(tokenizer)
            train_dataset = MarkdownDataset(train_df, tokenizer, fold, mode='train', parameter=parameter,
                                            max_seq_length=parameter.seq_length)
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=parameter.batch_size,
                                      num_workers=parameter.n_jobs, pin_memory=True, worker_init_fn=worker_init_fn,
                                      collate_fn=collate_fn)
            valid_dataset = MarkdownDataset(train_df, tokenizer, fold, mode='valid', parameter=parameter,
                                            max_seq_length=parameter.max_seq_length)
            valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=1,
                                      num_workers=parameter.n_jobs, pin_memory=True, collate_fn=collate_fn)
            criterion = MyLoss()  # MyBCELoss()  # MyLoss()
            best_score = -1
            patience_cnt = 0
            is_improved = True
            optimizer = AdamW(model.parameters(), lr=parameter.lr, betas=(0.9, 0.999), eps=1e-6,
                              weight_decay=args.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=parameter.base_epoch, eta_min=parameter.lr / 5)
            # scheduler = get_constant_schedule_with_warmup(optimizer, 100)
            # scheduler = get_cosine_schedule_with_warmup(optimizer, 100, num_training_steps= len(train_loader))
            # log.write('not loaded trained model\n')
            for epoch in range(parameter.base_epoch):
                # adjust learning rate for each epoch
                # adjust_learning_rate(optimizer, epoch, parameter)
                scheduler.step(epoch=epoch)
                log.write('Fold: [{0}] Epoch: [{1}], lr:[{2}]\n'.format(fold, epoch, optimizer.param_groups[0]['lr']))
                is_adversial = False
                train_loss, train_batch_time = train(model, train_loader, criterion, scheduler, optimizer, epoch,
                                                     args.n_accumulate, is_adversial)
                log.write('Epoch avg loss: {0}, Epoch cost time:{1} min\n'.format(train_loss, train_batch_time / 60))
                with torch.no_grad():
                    rank_pred, score = validate(model, valid_loader, code_df_valid)
                    log.write('Epoch score: {0}\n'.format(score))
                    if score > -1:  # best_score:
                        best_score, best_epoch = score, epoch
                        # oof_prediction[np.where(train_df['fold_flag'] == fold)] = rank_pred.copy()
                        save_model(model, model_dir, '{}_fold{}'.format(args.model_name, fold))
                        log.write('********Best Epoch: [{0}], Best Score:{1}********\n'.format(best_epoch, best_score))
                    else:
                        is_improved = False
                        patience_cnt += 1
        else:
            valid_dataset = MarkdownDataset(train_df, tokenizer, fold, mode='valid', parameter=parameter)
            valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=parameter.batch_size * 4,
                                      num_workers=parameter.n_jobs, pin_memory=True)
            model.load_state_dict(torch.load(train_model_filename, map_location='cpu')['state_dict'])
            model.cuda()
            log.write('loaded trained model\n')
            with torch.no_grad():
                rank_pred, score = validate(model, valid_loader, code_df_valid)
            log.write('Epoch score: {0}\n'.format(score))
            best_score = score
            # oof_prediction[np.where(train_df['fold_flag'] == fold)] = rank_pred.copy()
        # save_model(model, model_dir,'{}_fold{}'.format(args.model_name, fold))
        eval_loss.append(best_score)
        del model
        _ = gc.collect()
        torch.cuda.empty_cache()
    log.write('CV mean:{} std:{}.'.format(np.mean(eval_loss), np.std(eval_loss)))
    log.write('detail:{}'.format(np.round(eval_loss, 4)))
    np.save(os.path.join(parameter.result_dir, args.model_name + '_oof.npy'), oof_prediction)


def train(model, train_loader, criterion, scheduler, optimizer, epoch, n_accumulate=1, is_adversial=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    fgm = FGM(model) if is_adversial else None
    start = time.time()
    for i, batch_data in enumerate(train_loader):
        # optimizer.zero_grad()
        if parameter.use_cuda:
            batch_data = (t.cuda() for t in batch_data)
        seq, seq_mask, target_mask, target = batch_data
        # print(seq.shape,seq_mask.shape,target_mask.shape,target.shape)
        output = model(seq, seq_mask)
        loss = criterion(output, target, target_mask)
        losses.update(loss.item())
        loss = loss / n_accumulate
        loss.backward()
        if is_adversial:
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            output = model(seq, seq_mask)
            loss_adv = criterion(output, target, target_mask)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (i + 1) % n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
        # optimizer.step()
        # scheduler.step()

        batch_time.update(time.time() - start)
        start = time.time()
        if i % parameter.print_freq == 0:
            log.write('Epoch: [{0}][{1}/{2}], Loss {loss:.4f}\n'.format(epoch, i, len(train_loader), loss=loss.item()))
    return losses.avg, batch_time.sum


def validate(model, valid_loader, code_df_valid):
    # batch_time = AverageMeter()
    # losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    rank_pred = []
    masks = []
    for i, batch_data in enumerate(valid_loader):
        if parameter.use_cuda:
            batch_data = (t.cuda() for t in batch_data)
        seq, seq_mask, target_mask, _ = batch_data
        outputs = model(seq, seq_mask).detach().cpu().numpy()
        target_mask = target_mask.detach().cpu().numpy()
        tmp1 = np.zeros((outputs.shape[0], 4096))
        tmp1[:, :outputs.shape[1]] = outputs
        tmp2 = np.zeros((outputs.shape[0], 4096))
        tmp2[:, :outputs.shape[1]] = target_mask
        rank_pred.append(tmp1)
        masks.append(tmp2)
        # print(y_pred)
    rank_pred = np.concatenate(rank_pred)
    masks = np.concatenate(masks)
    # print(y_pred.shape)
    # y_pred = y_pred.reshape((-1, parameter.seq_length))
    score = get_score(valid_loader.dataset.meta_data.copy(), masks, rank_pred, code_df_valid)
    return rank_pred, score


if __name__ == '__main__':
    main()
