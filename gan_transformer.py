'''Defines the neural network, loss function and metrics'''
import math, copy
import numpy as np
import torch
import torch.nn as nn
from entmax import sparsemax, entmax15, entmax_bisect, EntmaxBisect
import torch.nn.functional as F
from torch.autograd import Variable
import utils

import logging
from torch.autograd import grad

from IPython import embed
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('Transformer.Net')

class EncoderDecoder(nn.Module):
    def __init__(self, params, emb, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.predict_steps = params.predict_steps

    def forward(self, x, idx):
        src_mask, encoder_out = self.encode(x[:,:-self.predict_steps,:], idx)
        #mu_en, sigma_en = self.generator(encoder_out)
        decoder_out = self.decode(encoder_out, x[:,-self.predict_steps:,:], idx, src_mask)
        q50, q90 = self.generator(decoder_out)
       
        #mu = torch.cat((mu_en, mu_de), 1)
        #sigma = torch.cat((sigma_en, sigma_de), 1)
        return q50, q90

    def encode(self, x, idx):
        src_mask = (x[:,:,0]!=0).unsqueeze(-2)
        src_mask1 = make_std_mask(x[:,:,0], 0)
        embeded = self.emb(x, idx)
        encoder_out = self.encoder(embeded, None)
        
        return src_mask, encoder_out

    def decode(self, memory, x, idx, src_mask):
        tgt_mask = make_std_mask(x[:,:,0], 0)
        embeded = self.emb(x, idx)
        decoder_out = self.decoder(embeded, memory, None, tgt_mask)
        return decoder_out

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embedding(nn.Module):
    def __init__(self, params, position):
        super(Embedding, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim) 
        # self.embed1 = nn.Linear(params.embedding_dim + params.cov_dim+ 1, params.d_model) #!!!!!!!parts is 24, others 25..........
        '''
        if(params.dataset == "wind"):
            self.embed1 = nn.Linear(6, params.d_model)
        else:
        '''
        self.embedding
        self.embed1 = nn.Linear(params.embedding_dim + params.cov_dim+ 1, params.d_model)
        self.embed2 = position
        
    def forward(self, x, idx):
        "Pass the input (and mask) through each layer in turn.  x : [bs, window_len, 5] "

        idx = idx.repeat(1, x.shape[1]) # idx is the store id of this batch , [bs, window_len]
        '''
        if(self.params.dataset=="wind"):
            idx = torch.unsqueeze(idx, -1)
            output = torch.cat((x, idx.float()), dim=2) # [bs, widow_len, 25]  [bs, window]  wind dataset!!!
        else:
        '''
        onehot_embed = self.embedding(idx) #[bs, windows_len, embedding_dim(default 20)] 
        try:
            output = torch.cat((x, onehot_embed), dim=-1)
            output = self.embed2(self.embed1(output))
        except:
            embed()
        return output
        
class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.q50 = nn.Linear(params.d_model, 1)
        self.q90 = nn.Linear(params.d_model, 1)
        
    def forward(self, x):
        q50 = self.q50(x)
        q90 = self.q90(x)
        return  torch.squeeze(q50),torch.squeeze(q90)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, params, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, params.N)
        self.norm = LayerNorm(layer.size)
         
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, src_mask)
        encoder_output = self.norm(x)
        return encoder_output
        
class EncoderLayer(nn.Module):
    def __init__(self, params, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.params = params
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(params.d_model, dropout), 2)
        self.size = params.d_model
    
    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)     

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, params, layer):
        super(Decoder, self).__init__()
        self.layers = clones(layer, params.N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, params, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = params.d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
        
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class AlphaChooser(torch.nn.Module):
    def __init__(self, head_count):
        """head_count (int): number of attention heads"""
        super(AlphaChooser, self).__init__()
        self.pre_alpha = nn.Parameter(torch.randn(head_count))

    def forward(self):
        alpha = 1 + torch.sigmoid(self.pre_alpha)
        return torch.clamp(alpha, min=1.01, max=2)

class EntmaxAlphaBencher(object):
    def __init__(self, X, alpha, n_iter=25):
        self.n_iter = n_iter
        self.X_data = X
        self.alpha = alpha

    def __enter__(self):
        self.X = self.X_data.clone().requires_grad_()
        self.dY = torch.randn_like(self.X)
        self.alpha = alpha
        return self

    def forward(self):
        self.Y = entmax_bisect(self.X, self.alpha, dim=-1, n_iter=self.n_iter)

    def backward(self):
        grad(outputs=(self.Y,),
             inputs=(self.X, self.alpha),
             grad_outputs=(self.Y))

    def __exit__(self, *args):
        try:
            del self.X
            del self.alpha
        except AttributeError:
            pass

        try:
            del self.Y
        except AttributeError:
            pass

def attention(query, key, value, params, mask=None, dropout=None, alpha=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9)
        except:
            embed()

    if params.attn_type=='softmax':
        p_attn = F.softmax(scores, dim = -1)
    elif params.attn_type=='sparsemax':
        p_attn = sparsemax(scores, dim=-1)
    elif params.attn_type=='entmax15':
        p_attn = entmax15(scores, dim=-1)
    elif params.attn_type=='entmax':
        p_attn = EntmaxBisect(scores, alpha, n_iter=25)
    else:
        raise Exception
    if dropout is not None:
        p_attn = dropout(p_attn)
    p_attn = p_attn.to(torch.float32)
    return torch.matmul(p_attn, value), scores, p_attn

class MultiHeadedAttention(nn.Module): 
    def __init__(self, params, dropout=0.2):  # TODO : h , dropout
        "Take in model size and number of heads." 
        super(MultiHeadedAttention, self).__init__()
        assert params.d_model % params.h == 0

        self.d_k = params.d_model // params.h
        self.h = params.h
        self.linears = clones(nn.Linear(params.d_model, params.d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.params = params
        self.scores = None
        self.alpha_choser = AlphaChooser(params.h)
        self.alpha = None
        self.attn_type = params.attn_type

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        if self.attn_type=='entmax':
            self.alpha = self.alpha_choser()
        x, self.scores, self.attn = attention(query, key, value, self.params, mask=mask, 
                                     dropout=self.dropout, alpha=self.alpha)
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
       

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=500): # TODO:max_len
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params.train_window, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

def test(model, params, x, v_batch, id_batch):
    batch_size = x.shape[0]
    sample_mu = torch.zeros(batch_size, params.predict_steps, device=params.device)
    sample_q90 = torch.zeros(batch_size, params.predict_steps, device=params.device)
    src_mask, memory = model.encode(x[:, :params.predict_start,:], id_batch)
    for t in range(params.predict_steps):
        ys = x[:, params.predict_start:params.predict_start+t+1,:]
        out = model.decode(memory, ys, id_batch, src_mask)
        q50, q90 = model.generator(out)
        if t!=0:    
            q50 = q50[:, -1]
            q90 = q90[:, -1]
        sample_mu[:, t] = q50 * v_batch[:, 0] + v_batch[:, 1]
        sample_q90[:, t] = q90* v_batch[:, 0]
        if t < (params.predict_steps - 1):
            x[:, params.predict_steps+t+1, 0] = q50

        return sample_mu, sample_q90


def loss_quantile(mu:Variable, labels:Variable, quantile:Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2*(torch.sum(quantile*((labels_e -mu_e)*I)+ (1-quantile) *(mu_e- labels_e)*(1-I)))
        loss += each_loss

    return loss

def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu_en: (Variable) dimension [batch_size, context_len] - estimated mean at time step t
        sigma_en: (Variable) dimension [batch_size, context_len] - estimated standard deviation at time step t
        mu_en: (Variable) dimension [batch_size, predict_len] - estimated mean at time step t
        sigma_en: (Variable) dimension [batch_size, predict_len] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    loss = 0
    zero_index = (labels != 0)
    for i in range(mu.shape[1]):
        zero_index = (labels[:, i] != 0)
        mu_e = mu[:, i]
        sigma_e = sigma[:, i]
        labels_e = labels[:, i]
        distribution = torch.distributions.normal.Normal(mu_e, sigma_e)
        likelihood = distribution.log_prob(labels_e)
        each_loss = -torch.mean(likelihood)
        loss += each_loss
    return loss

# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]
    
def accuracy_MAPE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return [diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    numerator = 0
    denominator = 0
    #pred_samples = samples.shape[0]
    rou_pred = mu
    abs_diff = labels - rou_pred
    numerator += 2 * (torch.sum(rou * abs_diff[labels > rou_pred]) - torch.sum(
        (1 - rou) * abs_diff[labels <= rou_pred])).item()
    denominator += torch.sum(labels).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def quantile_loss(quantile:float, mu:torch.Tensor, labels:torch.Tensor):
    #gaussian = torch.distributions.normal.Normal(mu, sigma)
    #pred = gaussian.sample()
    I = (labels >= mu).float()
    diff = 2*(torch.sum(quantile*((labels-mu)*I)+ (1-quantile) *(mu-labels)*(1-I))).item()
    denom = torch.sum(torch.abs(labels)).item()
    q_loss = diff/denom
    return q_loss

def MAPE(mu:torch.Tensor, labels:torch.Tensor):
    zero_index = (labels != 0)
    diff = mu[zero_index] -labels[zero_index]
    lo = torch.mean(torch.abs(diff / labels[zero_index])) *100
    return lo

def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != 0), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[mask] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result


def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    mask = labels == 0
    mu[mask] = 0.

    abs_diff = np.abs(labels - mu)
    abs_diff_1 = abs_diff.copy()
    abs_diff_1[labels < mu] = 0.
    abs_diff_2 = abs_diff.copy()
    abs_diff_2[labels >= mu] = 0.

    numerator = 2 * (rou * np.sum(abs_diff_1, axis=1) + (1 - rou) * np.sum(abs_diff_2, axis=1))
    denominator = np.sum(labels, axis=1)
    
    mask2 = (denominator == 0)
    denominator[mask2] = 1
    result = numerator / denominator
    result[mask2] = -1
    return result
