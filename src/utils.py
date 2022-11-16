import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import itertools


class KFold(object):
    """
    KFold: Group split by group_col or random_split
    """

    def __init__(self, random_seed, k_folds=10, flag_name='fold_flag'):
        self.k_folds = k_folds
        self.flag_name = flag_name
        np.random.seed(random_seed)

    def group_split(self, train_df, group_col):
        group_value = list(set(train_df[group_col]))
        group_value.sort()
        fold_flag = [i % self.k_folds for i in range(len(group_value))]
        np.random.shuffle(fold_flag)
        train_df = train_df.merge(pd.DataFrame({group_col: group_value, self.flag_name: fold_flag}), how='left',
                                  on=group_col)
        return train_df

    def random_split(self, train_df):
        fold_flag = [i % self.k_folds for i in range(len(train_df))]
        np.random.shuffle(fold_flag)
        train_df[self.flag_name] = fold_flag
        return train_df

    def stratified_split(self, train_df, group_col):
        train_df[self.flag_name] = 1
        train_df[self.flag_name] = train_df.groupby(by=[group_col])[self.flag_name].rank(ascending=True,
                                                                                         method='first').astype(int)
        train_df[self.flag_name] = train_df[self.flag_name].sample(frac=1.0).reset_index(drop=True)
        train_df[self.flag_name] = train_df[self.flag_name] % self.k_folds
        return train_df


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        #         torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, save_path, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, model_name + '.pth.tar')
    torch.save({'state_dict': model.state_dict(), }, filename)
    # if is_best:
    #     best_filename = os.path.join(save_path, model_name + '_best_model.pth.tar')
    #     shutil.copyfile(filename, best_filename)


def load_model(model, load_path, model_name):
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    filename = os.path.join(load_path, model_name + '.pth.tar')
    model.load_state_dict(torch.load(filename)['state_dict'])
    return model


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed every 10 epochs"""
    # lr = args.lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.3 ** (epoch // 10))


def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, rank, rank_mask):
        # loss = (inputs - rank) ** 2 * rank_mask
        loss = torch.abs(inputs - rank) * rank_mask
        loss = torch.sum(loss, dim=1) / torch.sum(rank_mask, dim=1)
        loss = loss.mean()
        return loss


class MyBCELoss(nn.Module):
    def __init__(self, class_weight=False):
        super().__init__()
        self.class_weight = class_weight

    def forward(self, inputs, targets, mask, sample_weight=None):
        # print(inputs)
        # inputs = inputs[:,:targets.shape[1]]
        bce1 = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction='none')
        bce2 = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction='none')
        bce = 1 * bce1 * targets + bce2 * (1 - targets)
        # mask = torch.where(targets >= 0, torch.ones_like(bce), torch.zeros_like(bce))
        bce = bce * mask
        # print(bce)
        #         if sample_weight is not None:
        #             bce = bce * sample_weight.unsqueeze(1)
        loss = torch.sum(bce, dim=1) / torch.sum(mask, dim=1)
        loss = loss.mean()
        return loss


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                # print(name, param)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / max(norm, 0.001)
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


from bisect import bisect


# from https://www.kaggle.com/code/ryanholbrook/competition-metric-kendall-tau-correlation
# Actually O(N^2), but fast in practice for our data
def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        assert len(gt) == len(pred)
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def get_score(df, masks, rank_pred, code_df_valid):
    #     df['cell_id2'] = [[y[i] for i in range(len(x)) if x[i] == 1] for x, y in
    #                           zip(df['cell_type'].values, df['cell_id'].values)]
    df['cell_id2'] = df['cell_id']
    df = df[['id', 'cell_id2']].explode('cell_id2')
    df = df[~pd.isnull(df['cell_id2'])]
    preds = rank_pred.flatten()[np.where(masks.flatten() == 1)]
    df['rank2'] = preds
    df = df.groupby(by=['id', 'cell_id2'], as_index=False)['rank2'].agg('mean')

    df.rename(columns={'cell_id2': 'cell_id'}, inplace=True)
    code_df_valid_tmp = code_df_valid[code_df_valid['id'].isin(df['id'])]
    code_df_valid_tmp['rank3'] = code_df_valid_tmp.groupby(by=['id'])['rank2'].rank(ascending=True, method='first')
    tmp = code_df_valid_tmp[['id', 'cell_id', 'rank3']].merge(df, how='inner', on=['id', 'cell_id'])
    tmp['rank4'] = tmp.groupby(by=['id'])['rank2'].rank(ascending=True, method='first')
    tmp = tmp[['id', 'cell_id', 'rank3']].merge(tmp[['id', 'rank4', 'rank2']].rename(columns={'rank4': 'rank3'}),
                                                how='inner', on=['id', 'rank3'])
    tmp = tmp[['id', 'cell_id', 'rank2']]

    df = df.merge(tmp[['id', 'cell_id', 'rank2']].rename(columns={'rank2': 'rank3'}), how='left', on=['id', 'cell_id'])
    df['rank2'] = np.where(pd.isnull(df['rank3']), df['rank2'], df['rank3'])

    # df = pd.concat([df[['id', 'cell_id', 'rank2']], code_df_valid_tmp]).reset_index(drop=True)
    df = df.sort_values(by=['id', 'rank2'], ascending=True)
    res = df.groupby(by=['id'], sort=False, as_index=False)['cell_id'].agg(list)

    train_orders = pd.read_csv('../input/AI4Code/train_orders.csv')
    train_orders['cell_order'] = train_orders['cell_order'].str.split()
    res = res.merge(train_orders, how='left', on='id')
    print(res)
    score = kendall_tau(res['cell_order'], res['cell_id'])
    return score


def get_model_path(model_name):
    res = '../input/'
    if model_name in ['distilroberta-base', 'roberta-base', 'roberta-large']:
        res += 'roberta-transformers-pytorch/' + model_name
    elif model_name in ['bart-base', 'bart-large']:
        res += 'bartbase' if model_name == 'bart-base' else 'bartlarge'
        res += '/'
    elif model_name in ['deberta-base', 'deberta-large', 'deberta-v2-xlarge', 'deberta-v2-xxlarge']:
        res += 'deberta/' + model_name.replace('deberta-', '')
    elif model_name in ['deberta-v3-large']:
        res += 'deberta-v3-large/' + model_name
    elif model_name in ['electra-base', 'electra-large']:
        res += 'electra/' + model_name + '-discriminator'
    elif 'albert' in model_name:
        res += 'pretrained-albert-pytorch/' + model_name
    elif model_name == 'deberta-v3-base':
        res += 'deberta-v3-base/' + model_name
    elif model_name == 'deberta-v3-large':
        res += 'deberta-v3-large/' + model_name
    elif model_name == 'funnel-large':
        res += 'funnel-large/'
    elif model_name == 'xlnet-base':
        res += 'xlnet-pretrained/xlnet-pretrained/'
    elif model_name == 'deberta-base-mnli':
        res += 'huggingface-deberta-variants/deberta-base-mnli/deberta-base-mnli/'
    elif model_name == 'deberta-xlarge':
        res += 'huggingface-deberta-variants/deberta-xlarge/deberta-xlarge/'
    elif model_name == 'codebert-base':
        res += 'codebert-base/codebert-base/'
    elif model_name == 'CodeBERTa-small-v1':
        res += 'huggingface-code-models/CodeBERTa-small-v1/'
    elif model_name == 'mdeberta-v3-base':
        res += 'mdeberta-v3-base/'
    else:
        raise ValueError(model_name)
    return res
