import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import RobertaModel, RobertaConfig, AutoConfig, AutoModel, AutoModelForMaskedLM



class MarkdownModel(nn.Module):
    def __init__(self, name, num_classes=1, seq_length=96, pretrained=True):
        super(MarkdownModel, self).__init__()
        # self.encoder = AutoModel.from_pretrained(name, attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
        self.config = AutoConfig.from_pretrained(name)
        self.config.attention_probs_dropout_prob = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.max_position_embeddings = 4096 * 2 
        # self.config.output_hidden_states = True
        if pretrained:
            self.encoder = AutoModel.from_pretrained(name, config=self.config, ignore_mismatched_sizes=True)
            # self.encoder = AutoModelForMaskedLM.from_pretrained(name, config=self.config)
        else:
            # self.encoder = AutoModelForMaskedLM.from_config(self.config)
            self.encoder = AutoModel.from_config(self.config)

        # self.encoder = AutoModel.from_pretrained(name)
        # print(self.encoder.__dict__)
        # transformer_layers = 2
#         self.seq_length = seq_length
#         self.transformer_layers = transformer_layers
        self.in_dim = self.encoder.config.hidden_size
        print(self.in_dim)
#         self.pe = PositionalEncoding(self.in_dim)
#         self.trans = nn.Sequential(
#             *[TransformerBlock(emb_s=64, head_cnt=self.in_dim // 64, dp1=0., dp2=0.) for _ in
#               range(transformer_layers)])
        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1, 
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        # self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
#         hidden = 64
#         dropout = 0.
#         self.sequence = nn.Sequential(
#             # nn.BatchNorm1d(1),
#             nn.Linear(1, hidden),  # todo
#             nn.Dropout(dropout),
#             nn.ReLU(),
#             # nn.BatchNorm1d(hidden),
#             nn.Linear(hidden, hidden),
#             nn.Dropout(dropout),
#             nn.ReLU()
#         )
        self.last_fc = nn.Linear(self.in_dim*2, num_classes)
        # self.fc = nn.LazyLinear(num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)
        self.sig = nn.Sigmoid()

    def forward(self, x, mask):
        x = self.encoder(x, attention_mask=mask)["last_hidden_state"]
        # x = x.reshape(-1, code_count, self.seq_length, self.in_dim).mean(2)
        #         x = torch.sum(x * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=1).unsqueeze(-1)
        #         x = x.reshape(-1, code_count, self.in_dim)
        # x = x + self.sequence(dense_features.unsqueeze(-1))
        # print(x)
        # print(x.shape)
#         prev = None
#         x = self.pe(x)
#         for i in range(self.transformer_layers):
#             # x = x * mask.unsqueeze(-1)
#             x, prev = self.trans[i](x, prev)
        # x = torch.sum(x * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=1).unsqueeze(-1)
        # x = torch.cat([x, self.sequence(dense_features.unsqueeze(1)).repeat(1,2048,1)], dim=2)
        # x = x.mean(1)
        x, _ = self.bilstm(x)
        out = self.last_fc(x)
#         for i, dropout in enumerate(self.dropouts):
#             if i == 0:
#                 out = self.last_fc(dropout(x))
#             else:
#                 out += self.last_fc(dropout(x))
#         out /= len(self.dropouts)
        # out = self.sig(out)
        out = out.squeeze(-1)
        return out
# input = torch.randn(2, 200).long() +10
# input2 = torch.zeros(2, 200)
# net = MarkdownModel('roberta-base', pretrained=False)
# print(input)
# print(net(input, input2))
