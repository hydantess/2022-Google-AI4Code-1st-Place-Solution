import torch
import torch.nn as nn
import argparse
import os
import time
import random
import numpy as np
from utils import get_model_path, seed_everything, save_model
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup,AutoModel
from typing import Tuple, Optional
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss, MSELoss
from torch.cuda.amp import autocast, GradScaler
from parameter import Parameter
from data_processing import preprocess_df
import pandas as pd

parameter = Parameter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join(parameter.result_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

class MlmData(Dataset):

    def __init__(self, data, max_length, tokenizer):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data = data['source'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        encoded = self.tokenizer.encode_plus(
            text=data,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=4096,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
            truncation=True,
            return_special_tokens_mask=True
        )
        
        inputs, special_tokens_mask, attention_mask= encoded['input_ids'], encoded['special_tokens_mask'], encoded['attention_mask']
        input_size = attention_mask.sum()
        if input_size >self.max_length:
            start_index = random.choice(range(input_size-self.max_length))
            inputs = inputs[:,start_index:start_index + self.max_length]
            special_tokens_mask = special_tokens_mask[:, start_index:start_index + self.max_length]
            attention_mask = attention_mask[:, start_index:start_index + self.max_length]
        else:
            inputs = inputs[:,:self.max_length]
            special_tokens_mask = special_tokens_mask[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        
        inputs, labels = mask_tokens(inputs, special_tokens_mask,
                                     tokenizer=self.tokenizer)
        inputs, labels, attention_mask= inputs[0], labels[0], attention_mask[0]
        
            
#         inputs, labels = mask_tokens(encoded['input_ids'], encoded['special_tokens_mask'],
#                                      tokenizer=self.tokenizer)
#         inputs, labels, attention_mask= inputs[0], labels[0], encoded['attention_mask'][0]
#         input_size = attention_mask.sum()
#         if input_size >self.max_length:
#             start_index = random.choice(range(input_size-self.max_length))
#             inputs = inputs[start_index:start_index + self.max_length]
#             labels = labels[start_index:start_index + self.max_length]
#             attention_mask = attention_mask[start_index:start_index + self.max_length]
#         else:
#             inputs = inputs[:self.max_length]
#             labels = labels[:self.max_length]
#             attention_mask = attention_mask[:self.max_length]
            
        return inputs, labels, attention_mask


# def save_model(model, model_name):
#     output_dir = parameter.result_dir
#     # Create output directory if needed
#     model_out_dir = os.path.join(output_dir, 'models', model_name)
#     if not os.path.exists(model_out_dir):
#         os.makedirs(model_out_dir)
#     print("Saving model to %s: %s" % (model_out_dir, model_name))
#     # torch.save(model_to_save.state_dict(), output_dir+model_name)
#     model.save_pretrained(model_out_dir)
#     # tokenizer.save_vocabulary(model_out_dir)


class MlmModel(nn.Module):
    def __init__(self, model_name):
        super(MlmModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # self.config.output_hidden_states = True
        self.config.max_position_embeddings = 4096 * 2 
#         self.encoder = AutoModelForMaskedLM.from_config(model_name, config=self.config).to(device)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config, ignore_mismatched_sizes=True).to(device)
        self.fc = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False).to(device)

    def forward(self, inputs, labels, masks):
        outputs = self.encoder(inputs, attention_mask=masks.float())['last_hidden_state']
        # print(outputs)
        prediction_scores = self.fc(outputs)
        # print(inputs.size(), labels.size(), masks.size())
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss


def mask_tokens(inputs: torch.Tensor, special_tokens_mask: torch.Tensor, tokenizer=None,
                mlm_probability=0.15, max_predictions_per_seq=5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # special_tokens_mask = tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
    #     num_tokens = parameter.seq_length - special_tokens_mask.sum()
    #     num_to_mask = min(max_predictions_per_seq, max(1, int(num_tokens * mlm_probability)))

    # corpus_mask = (corpus_mask > 2.2).bool()
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(model_name, max_length, num_epochs, learning_rate, batch_size, n_accumulate):
    # load tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # load model
    # model = MlmModel(model_name)  # BertForMaskedLM.from_pretrained('bert-base-uncased')
    # model.resize_token_embeddings(len(tokenizer))

    # train params
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 4}

    # test_params = {"batch_size": batch_size,
    #                "shuffle": True,
    #                "num_workers": 0}

    print("==========Loading Data===========")

    train_df = pd.read_pickle('../input/ai4codetrainpicklefile/train_df.pkl')
    train_df = preprocess_df(train_df)
    # train_df['source'] = train_df['source'].apply(lambda x:x[:1000])
    train_df = pd.concat(
        [train_df[train_df['cell_type'] == 0], train_df[train_df['cell_type'] == 1].sample(frac=1.0)]).reset_index(
        drop=True)
    train_df = train_df.groupby(by=['id'],as_index=False)['source'].agg(lambda x: '[SEP]'.join(x))
    # train_df = train_df.iloc[:100]
    
#     if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#         return 1
    # patient_notes = patient_notes[~patient_notes['pn_history'].isin(train_df['pn_history'])]
    tokenizer = AutoTokenizer.from_pretrained(get_model_path(args.model_name))
    # tokenizer = AutoTokenizer.from_pretrained(get_model_path(model_name))
    # config = AutoConfig.from_pretrained(get_model_path(model_name))
    model = MlmModel(get_model_path(model_name))
    # filename = './user_data/models/deberta-v3-large_pre_5.pth.tar'
    # model.encoder.load_state_dict(torch.load(filename)['state_dict'])
    # model.encoder.resize_token_embeddings(len(tokenizer))
    # model = AutoModelForMaskedLM.from_pretrained(get_model_path(model_name),config=config).to(device)
    # model.resize_token_embeddings(len(tokenizer))
    # model.to(device)
    train_set = MlmData(train_df, max_length, tokenizer)
    # val_set = MlmData(validate, max_length, tokenizer)
    # save_model(model.encoder, model_dir, model_name + '_pre')
    train_loader = DataLoader(train_set, **training_params)
    # val_loader = DataLoader(val_set, **test_params)
    print("==========Data Loaded===========")

    # params
    num_training_steps = len(train_loader) * num_epochs  # batches into the number of training epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    train_loss, val_loss = [], []
    best_perplexity = 100
    scaler = GradScaler() 

    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0
        total_val_loss = 0
        t1 = time.time()
        # if epoch<=4:
        #     continue

        print("------------Start Training------------")

        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        for iter, batch in enumerate(train_loader):
            # batch = (t.type(torch.LongTensor).to(device) for t in batch)
            #inputs, labels, masks = batch
            inputs, labels, masks = (t.type(torch.LongTensor).to(device) for t in batch)
            # print(inputs.shape,labels.shape,masks.shape)
            # optimizer.zero_grad()
            loss = model(inputs, labels, masks)
#             with autocast():
#                 loss = model(inputs, labels, masks)
                # scaler.scale(loss).backward()
            # output = model(inputs, masks, labels=labels)
            # loss, logits = output[:2]
    
            # loss = model(inputs, labels, masks)
            # loss, logits = output[:2]
            # total_train_loss += loss.item()
            if (iter+1) % 500 == 0:
                print("Train Batch: {}/{} --- Train Loss: {}".format(iter+1, len(train_loader),loss.item()))

            loss.backward()

            # Clip the norm of the gradients to 10
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if (iter + 1) % n_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             optimizer.step()
#             scheduler.step()
        print("cost time: {} min ".format((time.time() - t1) / 60 ))
        if (epoch+1) % 5 == 0:
            save_model(model.encoder, model_dir, model_name + '_pre_{}'.format(epoch + 1))
    #         mean_loss = total_train_loss / len(train_loader)
    #         train_loss.append(mean_loss)
    # save_model(model.encoder, model_dir, model_name + '_pre')

    #     print("------------Validate Model------------")
    #     model.eval()
    #
    #     for iter, batch in enumerate(val_loader):
    #         batch = (t.type(torch.LongTensor).to(device) for t in batch)
    #         inputs, attn, labels = batch
    #
    #         with torch.no_grad():
    #             output = model(inputs.squeeze(dim=1), attn.squeeze(dim=1), labels=labels.squeeze(dim=1))
    #             loss = output[0]
    #             total_val_loss += loss.item()
    #
    #             print("Validation Batch: {} --- Validation Loss: {}".format(iter + 1, loss))
    #
    #     mean_loss = total_val_loss / len(val_loader)
    #     perplexity = math.exp(mean_loss)
    #     print("Perplexity: {}".format(perplexity))
    #     val_loss.append(mean_loss)
    #
    #     if perplexity < best_perplexity:
    #         best_perplexity = perplexity
    #         print('----------Saving model-----------')
    #         save_model(model=model, model_name='test_bert.pt', tokenizer=tokenizer)
    #         print('----------Model saved-----------')
    #
    # print('----------Training Complete-----------')


if __name__ == "__main__":
    seed_everything(parameter.random_seed)
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--out_dir', default='./user_data', type=str,
                        help='destination where trained network should be saved')
    parser.add_argument('--model_name', default='roberta-base', type=str)
    parser.add_argument('--base_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_length', default=2048, type=int)
    parser.add_argument('--n_accumulate', default=1, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    args = parser.parse_args()
    model_out_dir = os.path.join(parameter.result_dir, 'models')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    train(args.model_name, max_length=args.max_length, num_epochs=args.base_epoch, learning_rate=args.learning_rate,
          batch_size=args.batch_size,n_accumulate=args.n_accumulate)
