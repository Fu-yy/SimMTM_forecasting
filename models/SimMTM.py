import argparse

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding, Patch
from utils.losses import AutomaticWeightedLoss
from utils.tools import ContrastiveWeight, AggregationRebuild

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
        return x

class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()

        pn = seq_len * d_model
        dimension = 128

        self.flat = nn.Flatten(start_dim=-2)
        self.line1 = nn.Linear(pn, pn // 2)
        self.line2 = nn.Linear(pn // 2, dimension)
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        # f = self.flat(x)
        # x = self.line1(f)
        # x = self.line2(x)


        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class Model(nn.Module):
    """
    Transformer with channel independent + SimMTM
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.manifolds_weights = configs.manifolds_weights
        weights = [0.5, 0.3, 0.2]  # 示例权重，根据实际情况调整

        # Embedding
        self.enc_embedding = DataEmbedding(self.patch_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Patch

        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )
        # Decoder
        if self.task_name == 'pretrain':
            # for reconstruction

            inner_seq_len = configs.seq_len // self.patch_len
            # self.projection = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
            self.projection = Flatten_Head(inner_seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)

            # for series-wise representation
            self.pooler = Pooler_Head(inner_seq_len, configs.d_model, head_dropout=configs.head_dropout)

            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(self.configs)
            self.aggregation = AggregationRebuild(self.configs)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'finetune':
            inner_seq_len = configs.seq_len // self.patch_len

            self.head = Flatten_Head(inner_seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
            # self.head = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)


    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.unsqueeze(-1) # x_enc: [bs x n_vars x seq_len x 1]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]


        x_patch = self.patch(x_enc)  # [batch_size * num_features, seq_len, patch_len]

        # embedding
        enc_out = self.enc_embedding(x_patch) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        # point-wise representation
        p_enc_out, attns = self.encoder(enc_out) # p_enc_out: [(bs * n_vars) x seq_len x d_model]

        total_loss_cl = 0.0
        for i,fold_item_weight in enumerate(self.manifolds_weights):
            # series-wise representation
            s_enc_out = self.pooler(p_enc_out) # s_enc_out: [(bs * n_vars) x dimension]
            # series weight learning
            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out) # similarity_matrix: [(bs * n_vars) x (bs * n_vars)]
            rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) # agg_enc_out: [(bs * n_vars) x seq_len x d_model]
            p_enc_out = agg_enc_out
            # total_loss_cl += fold_item_weight * loss_cl
            total_loss_cl +=  loss_cl
        total_loss_cl = total_loss_cl / len(self.manifolds_weights)
        agg_enc_out = agg_enc_out.reshape(bs, n_vars, seq_len, -1) # agg_enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.projection(agg_enc_out)  # dec_out: [bs x n_vars x seq_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x seq_len x n_vars]

        # de-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        pred_batch_x = dec_out[:batch_x.shape[0]]

        # series reconstruction
        loss_rb = self.mse(pred_batch_x, batch_x.detach())

        # loss
        loss = self.awl(total_loss_cl, loss_rb)

        return loss, total_loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x


    def forecast(self, x_enc, x_mark_enc):

        # data shape
        bs, seq_len, n_vars = x_enc.shape

        # normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # channel independent
        x_enc = x_enc.permute(0, 2, 1) # x_enc: [bs x n_vars x seq_len]
        x_enc = x_enc.reshape(-1, seq_len, 1) # x_enc: [(bs * n_vars) x seq_len x 1]

        x_patch = self.patch(x_enc)  # [batch_size * num_features, seq_len, patch_len]

        # embedding
        #x_mark_enc = torch.repeat_interleave(x_mark_enc, repeats=n_vars, dim=0)
        #enc_out = self.enc_embedding(enc_out, x_mark_enc)
        enc_out = self.enc_embedding(x_patch) # enc_out: [(bs * n_vars) x seq_len x d_model]

        # encoder
        enc_out, attns = self.encoder(enc_out) # enc_out: [(bs * n_vars) x seq_len x d_model]

        enc_out = torch.reshape(enc_out, (bs, n_vars, seq_len, -1)) # enc_out: [bs x n_vars x seq_len x d_model]

        # decoder
        dec_out = self.head(enc_out)  # dec_out: [bs x n_vars x pred_len]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x pred_len x n_vars]

        # de-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, batch_x=None, mask=None):

        if self.task_name == 'pretrain':
            return self.pretrain(x_enc, x_mark_enc, batch_x, mask)
        if self.task_name == 'finetune':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None


def get_config():
    parser = argparse.ArgumentParser(description='SimMTM')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='pretrain',
                        help='task name, options:[pretrain, finetune]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='SimMTM', help='model id')
    parser.add_argument('--model', type=str, required=False, default='SimMTM', help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./outputs/checkpoints/',
                        help='location of model fine-tuning checkpoints')
    parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/',
                        help='location of model pre-training checkpoints')
    parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best.pth',
                        help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
    parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
    parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--patch_len', type=int, default=12, help='path length')
    parser.add_argument('--stride', type=int, default=12, help='stride')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # Pre-train
    parser.add_argument('--lm', type=int, default=3, help='average masking length')
    parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
    parser.add_argument('--rbtp', type=int, default=1,
                        help='0: rebuild the embedding of oral series; 1: rebuild oral series')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument('--masked_rule', type=str, default='geometric',
                        help='geometric, random, masked tail, masked head')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--time_steps', type=int, default=1000, help='time_steps time_steps')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--input_len', type=int,default=336, help='input_len')

    parser.add_argument('--manifolds_weights', type=float,default=[0.33,0.33,0.33], help='manifolds_weights')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = get_config()
    config.task_name = "pretrain"
    config.root_path ="./ dataset / ETT - small /"
    config.data_path = "ETTh2.csv"
    config.model_id = "ETTh2"
    config.model = "SimMTM"
    config.data = "ETTh2"
    config.features = "M"
    config.seq_len = 336
    config.e_layers = 2
    config.enc_in = 7
    config.dec_in = 7
    config.c_out = 7
    config.n_heads = 8
    config.d_model = 8
    config.d_ff = 32
    config.positive_nums = 1
    config.mask_rate = 0.5
    config.learning_rate = 0.001
    config.batch_size = 16
    config.train_epochs = 20
    config.use_gpu = 1
    config.patch_len = 2
    config.stride = 2

    model = Model(config)
    x = torch.randn(config.batch_size,336,7)
    x_mask = torch.randn((config.positive_nums  + 1) * config.batch_size,336,7)
    x_mask_enc = torch.randn((config.positive_nums  + 1) * config.batch_size,336,4)
    mask = torch.ones_like(x_mask)

    res = model( x_mask, x_mask_enc, x,mask)

    c = 'end'

