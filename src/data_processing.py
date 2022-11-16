import pandas as pd
import numpy as np
import random
import gc
import time
import shutil
import re
import os
from tqdm import tqdm
import glob
# from unidecode import unidecode
from parameter import Parameter
from utils import KFold

parameter = Parameter()


def get_data(seed=27, mode=0):
    if os.path.exists('../input/ai4codetrainpicklefile/train_df.pkl'):
        train_df = pd.read_pickle('../input/ai4codetrainpicklefile/train_df.pkl')
    else:
        train_df = read_json_data(mode='train')
        train_orders = pd.read_csv('../input/AI4Code/' + 'train_orders.csv')
        train_ancestors = pd.read_csv('../input/AI4Code/' + 'train_ancestors.csv')

        train_orders['cell_id'] = train_orders['cell_order'].str.split()
        train_orders = train_orders.explode(column='cell_id')
        train_orders['flag'] = range(len(train_orders))
        train_orders['rank'] = train_orders.groupby(by=['id'])['flag'].rank(ascending=True, method='first').astype(int)
        del train_orders['flag'], train_orders['cell_order']
        print(train_orders)
        # train_df = preprocess_features(train_df)
        train_df = train_df.merge(train_orders, on=['id', 'cell_id'], how='left')
        train_df = train_df.merge(train_ancestors[['id', 'ancestor_id']], on=['id'], how='left')
        train_df.to_pickle('train_df.pkl')

    train_df = KFold(seed, parameter.k_folds).group_split(train_df, group_col='ancestor_id')
    # train_df = preprocess_features(train_df)
    # train_df['source_length'] = train_df['source'].apply(len)
    # train_df['id_length'] = train_df.groupby(by=['id'])['source_length'].transform('sum')
    train_df = preprocess_df(train_df)
    train_df = pd.concat(
        [train_df[train_df['cell_type'] == 0], train_df[train_df['cell_type'] == 1].sample(frac=1.0)]).reset_index(
        drop=True)
    train_df['rank2'] = (train_df.groupby(by=['id', 'cell_type']).cumcount() + 1) / \
                        train_df.groupby(by=['id', 'cell_type'])['cell_id'].transform('count')
    train_df.loc[train_df['cell_type'] == 1, 'rank2'] = -1
    code_df_valid = train_df[train_df['cell_type'] == 0][['id', 'cell_id', 'rank2']].copy()

    #     for col in ['cell_count','markdown_count', 'code_count']:
    #         train_df[col] = (train_df[col] - train_df[col].mean())/ train_df[col].std()
    #         train_df[col] = np.clip(train_df[col].fillna(0.0), -3, 3)

    train_df = get_truncated_df(train_df, cell_count=parameter.cell_count)
    #     train_df['flag'] = train_df['cell_type'].apply(lambda x:np.sum(x))
    #     train_df = train_df[train_df['flag']>0]
    #     del train_df['flag']
    print(train_df)
    print(train_df.shape)
    return train_df, code_df_valid


def read_json_data(mode='train'):
    paths_train = sorted(list(glob.glob(parameter.data_dir + '{}/*.json'.format(mode))))  # [:100]
    res = pd.concat([
        pd.read_json(path, dtype={'cell_type': 'category', 'source': 'str'}).assign(
            id=path.split('/')[-1].split('.')[0]).rename_axis('cell_id')
        for path in tqdm(paths_train)]).reset_index(drop=False)
    res = res[['id', 'cell_id', 'cell_type', 'source']]
    return res


def preprocess_df(df):
    df['cell_count'] = df.groupby(by=['id'])['cell_id'].transform('count')
    # df['source'] = df['cell_type'] + ' ' + df['source']
    df['cell_type'] = df['cell_type'].map({'code': 0, 'markdown': 1}).fillna(0).astype(int)
    # df.loc[df['cell_type']==0, 'source'] = df.loc[df['cell_type']==0, 'rank'] + ' ' + df.loc[df['cell_type']==0, 'source']
    df['markdown_count'] = df.groupby(by=['id'])['cell_type'].transform('sum')
    df['code_count'] = df['cell_count'] - df['markdown_count']
    df['rank'] = df['rank'] / df['cell_count']
    df['source'] = df['source'].apply(lambda x: x.lower().strip())
    df['source'] = df['source'].apply(lambda x: preprocess_text(x))
    # df['source'] = df['source'].replace("\\n", "\n")
    # df['source'] = df['source'].str.replace("\n", "")
    df['source'] = df['source'].str.replace("[SEP]", "")
    df['source'] = df['source'].str.replace("[CLS]", "")

    # df['source'] = df['source'].replace("#", "")
    # df['source'] = df['source'].apply(lambda x: unidecode(x))
    df['source'] = df['source'].apply(lambda x: re.sub(' +', ' ', x))
    return df


# from https://www.kaggle.com/code/ilyaryabov/fastttext-sorting-with-cosine-distance-algo
import re
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()


def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    document = document.replace('_', ' ')

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    #         # Remove single characters from the start
    #         document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    #         # Removing prefixed 'b'
    #         document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()
    # return document

    #         # Lemmatization
    #         tokens = document.split()
    #         tokens = [stemmer.lemmatize(word) for word in tokens]
    #         # tokens = [word for word in tokens if len(word) > 3]

    #         preprocessed_text = ' '.join(tokens)
    return document


# def get_truncated_df(df, cell_count=128, id_col='id2', group_col='id', max_random_cnt=500, expand_ratio=10):
#     tmp1 = df[df['cell_count'] <= cell_count].reset_index(drop=True)
#     tmp1.loc[:, id_col] = 0
#     tmp2 = df[df['cell_count'] > cell_count].reset_index(drop=True)
#     # print(tmp1.shape,tmp2.shape)
#     res = [tmp1]
#     for _, df_g in tmp2.groupby(by=group_col):
#         # print(df_g.columns)
#         df_g = df_g.sample(frac=1.0).reset_index(drop=True)
#         # index_list = range(len(df_g))
#         step = min(cell_count // 2, len(df_g) - cell_count)
#         step = max(step, 1)
#         id_col_count = 0
#         for i in range(0, len(df_g), step):
#             # indexes = [i] + list(np.random.choice([j for j in index_list if j!=i],cell_count-1, replace=False))
#             # indexes = range(i,i+cell_count)
#             # print(indexes,i,len(df_g),index_list)
#             res_tmp = df_g.iloc[i:i + cell_count]  # .copy()
#             # if len(res_tmp) == cell_count:
#             res_tmp.loc[:, id_col] = id_col_count
#             id_col_count += 1
#             res.append(res_tmp)
#
#         random_cnt = int(len(df_g) // cell_count * expand_ratio)
#         random_cnt = min(random_cnt, max_random_cnt)  # todo
#         if random_cnt > 0:
#             for i in range(random_cnt):
#                 res_tmp = df_g.sample(n=cell_count).reset_index(drop=True)
#                 res_tmp.loc[:, id_col] = id_col_count
#                 id_col_count += 1
#                 res.append(res_tmp)
#
#     res = pd.concat(res).reset_index(drop=True)
#     sort_flag = range(len(res))
#     np.random.shuffle(sort_flag)
#     res.loc[res['cell_type'] == 0, 'sort_flag'] = 0
#     res = res.sort_values(by=['id', id_col, 'cell_type', 'rank'], ascending=True)
#     res = res.groupby(by=['id', id_col, 'fold_flag', 'cell_count'], as_index=False, sort=False)[
#         ['cell_id', 'cell_type', 'source', 'rank']].agg(list)
#     return res


def get_truncated_df(df, cell_count=128, id_col='id2', group_col='id', max_random_cnt=100, expand_ratio=5):
    tmp1 = df[df['cell_count'] <= cell_count].reset_index(drop=True)
    tmp1.loc[:, id_col] = 1
    tmp2 = df[df['cell_count'] > cell_count].reset_index(drop=True)
    # print(tmp1.shape,tmp2.shape)
    res = [tmp1]
    for _, df_g in tmp2.groupby(by=group_col):
        # print(df_g.columns)
        df_g = df_g.sample(frac=1.0).reset_index(drop=True)
        step = min(cell_count // 2, len(df_g) - cell_count)
        step = max(step, 1)
        id_col_count = 1
        for i in range(0, len(df_g), step):
            res_tmp = df_g.iloc[i:i + cell_count]  # .copy()
            if len(res_tmp) != cell_count:
                res_tmp = df_g.iloc[-cell_count:]
            # if len(res_tmp) == cell_count:
            res_tmp.loc[:, id_col] = id_col_count
            id_col_count += 1
            res.append(res_tmp)
            if i + cell_count >= len(df_g):
                break

        if len(df_g) // cell_count > 1.3:
            random_cnt = int(len(df_g) // cell_count * expand_ratio)
            random_cnt = min(random_cnt, max_random_cnt)  # todo

            for i in range(random_cnt):
                res_tmp = df_g.sample(n=cell_count).reset_index(drop=True)
                res_tmp.loc[:, id_col] = id_col_count
                id_col_count += 1
                res.append(res_tmp)

    res = pd.concat(res).reset_index(drop=True)
    res = res.sort_values(by=['id', id_col, 'cell_type', 'rank2'], ascending=True)
    res = res.groupby(by=['id', id_col, 'fold_flag', 'cell_count', 'markdown_count', 'code_count'], as_index=False,
                      sort=False)[
        ['cell_id', 'cell_type', 'source', 'rank', 'rank2']].agg(list)
    return res
