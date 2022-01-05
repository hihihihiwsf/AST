import json
import logging
import os
import shutil

import torch
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy

import matplotlib
matplotlib.use('Agg')

#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions
import matplotlib.pyplot as plt

import seaborn as sns

import gan_transformer as transformer 

logger = logging.getLogger('Transformer.Utils')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        try:
            score = -val_loss
        except:
            embed()
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


class RunningAverage:
    '''A simple class that maintains the running average of a quantity
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('Transformer')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_prediction(prediction, path):
    
    index = prediction[:,0]
    result = pd.DataFrame(prediction[:,1], index=index)
    result.to_csv(path)

def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()

def save_loss(variable, save_name, location='./loss/'):
    #num_samples = variable.shape[0]
    path = os.path.join(location, save_name+'.csv')
    np.savetxt(path, variable, delimiter=',')

def plot_attn(attn, t, save_name, location='./figures/'):   
    at = attn.detach().cpu().numpy()
    for i in range(at.shape[1]): # h head
        #ax = sns.heatmap(at[-1][i], vmin=0, vmax=0.1, xticklabels=False, yticklabels=False)
        ax = sns.heatmap(at[-1][i], xticklabels=False, yticklabels=False)
        f = ax.get_figure()
        f.savefig(os.path.join(location, str(t) + save_name + str(i)+ '-th_head_attention.png'))
        f.clf()

def JS_div(p ,q):
    M = (p+q)/2
    JS = 0.5*scipy.stats.entropy(p,M) + 0.5*scipy.stats.entropy(q,M)
    return JS

def count_num(attn_score):
    div = np.zeros((10))
    #v = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,0.20, 0.4, 0.6, 0.8, 1.0]
    v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0]
    for i in range(len(v)):
        flag1 = attn_score <= v[i]
        if i>0: 
            flag2 = attn_score > (v[i-1])
        else:
            flag2 =  attn_score > 0
        div[i] = np.sum(np.array(flag1&flag2))
    return div, v

def plot_prediction():
    history = np.load("labels.npy")
    t_res = np.load("T_prediction.npy")
    st_res = np.load("ST_prediction.npy")
    at_res = np.load("AT_prediction.npy")
    his = history[4]
    t_re = t_res[4]
    st_re = st_res[4]
    at_re = at_res[4]
    x = np.arange(his.shape[0])
    embed()
    plt.figure()
    plt.xlim(0, 200)
    plt.ylim(1000, 4500)
    plt.vlines(168,1000,4500, color="green", linestyle='--')
    plt.plot(x, his, color='blue')
    plt.plot(x[168:], st_re, color='red', linestyle='--')
    plt.savefig(fname="prediction_ST.pdf")


def report_num(output, index, name):
    path = './attn'
    if not os.path.isdir(path):
        os.makedirs(path)
    save_path = os.path.join(path, str(name).split('/')[-1] +'_attn_number.csv')
    
    data = pd.DataFrame(output, columns=index)
    data.to_csv(save_path)
    

def init_metrics(sample=True):
    metrics = {
        'ND': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(3),  # numerator, denominator, time step count
        'test_loss': np.zeros(2),
        'Q50': np.zeros(2),
        'Q90': np.zeros(2),
        'q50': np.zeros(2),
        'q90': np.zeros(2),
        'MAPE':np.zeros(1)
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics

def get_metrics(sample_mu, labels, predict_start, samples=None, relative=False):
    metric = dict()
    metric['ND'] = transformer.accuracy_ND_(sample_mu, labels[:, predict_start:], relative=relative)
    metric['RMSE'] = transformer.accuracy_RMSE_(sample_mu, labels[:, predict_start:], relative=relative)
    metric['Q90'] = transformer.accuracy_ROU_(0.9, sample_mu, labels[:,predict_start:], relative=relative)
    metric['Q50'] = transformer.accuracy_ROU_(0.5, sample_mu, labels[:, predict_start:], relative=relative)
    #metric['q50'] = transformer.quantile_loss(0.5, sample_mu, labels[:, predict_start:])
    #metric['q90'] = transformer.quantile_loss(0.9, sample_mu, labels[:, predict_start:])
    if samples is not None:
        metric['rou90'] = transformer.accuracy_ROU_(0.9, samples, labels[:, predict_start:], relative=relative)
        metric['rou50'] = transformer.accuracy_ROU_(0.5, samples, labels[:, predict_start:], relative=relative)
    return metric

def update_metrics(raw_metrics, sample_mu, labels, predict_start, samples=None, relative=False):   #  the original update!!!!! 
    raw_metrics['ND'] = raw_metrics['ND'] + transformer.accuracy_ND(sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['RMSE'] = raw_metrics['RMSE'] + transformer.accuracy_RMSE(sample_mu, labels[:, predict_start:], relative=relative)
    input_time_steps = sample_mu.numel()
    raw_metrics['test_loss'] = raw_metrics['test_loss'] + [
        transformer.loss_quantile(sample_mu,labels, 0.5) * input_time_steps, input_time_steps]
    raw_metrics['Q90'] = raw_metrics['Q90'] + transformer.accuracy_ROU(0.9, sample_mu, labels[:, predict_start:], relative=relative)
    raw_metrics['Q50'] = raw_metrics['Q50'] + transformer.accuracy_ROU(0.5, sample_mu, labels[:, predict_start:], relative=relative)
    #raw_metrics['q50'] = raw_metrics['q50'] + transformer.quantile_loss(0.5, sample_mu, labels[:, predict_start:])
    #raw_metrics['q90'] = raw_metrics['q90'] + transformer.quantile_loss(0.9, sample_mu, labels[:, predict_start:])
    if samples is not None:
        raw_metrics['rou90'] = raw_metrics['rou90'] + transformer.accuracy_ROU(0.9, samples, labels[:, predict_start:], relative=relative)
        raw_metrics['rou50'] = raw_metrics['rou50'] + transformer.accuracy_ROU(0.5, samples, labels[:, predict_start:], relative=relative)
    return raw_metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
    summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][2]) / (
                raw_metrics['RMSE'][1] / raw_metrics['RMSE'][2])
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    summary_metric['Q90'] = (raw_metrics['Q90'][0] / raw_metrics['Q90'][1])
    summary_metric['Q50'] = (raw_metrics['Q50'][0] / raw_metrics['Q50'][1])
    summary_metric['q50'] = raw_metrics['q50']
    summary_metric['q90'] = raw_metrics['q90']
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]
    return summary_metric

def tSNE(model):
    for name, params in model.named_parameters():
        if isinstance(layer[1], transformer.Generator):
            embed()
            
