import argparse
import logging
import os

import numpy as np
import math, copy, time
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import json


import utils
from utils import EarlyStopping
import gan_transformer as transformer
from evaluate import evaluate
from opt import OpenAIAdam
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('Transformer.Train')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='elect', help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='transformermodify_model', help='Directory containing params.json')
parser.add_argument('--attn_transform', default='constrained_sparsemax', help='Parent dir of the dataset')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--gan', default="True", help="Whether to train adversarially")  
parser.add_argument('--dropout', type=float, default=0.01)  
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None, help='Optional, name of the file in --model_dir containing weights to reload before training')


def train(model: nn.Module,
          discriminator:nn.Module,
          optimizer_G,
          optimizer_D,
          adversarial_loss,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        discriminator: (torch.nn.Module) the discriminator network
        optimizer: (torch.optim) optimizer for parameters of model
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    d_loss_epoch = np.zeros(len(train_loader))
    e_loss_epoch = np.zeros(len(train_loader))

    for i, (train_batch, idx, labels_batch) in enumerate(tqdm(train_loader)):
        batch_size = train_batch.shape[0]

        train_batch = train_batch.to(torch.float32).to(params.device)  
        labels_batch = labels_batch.to(torch.float32).to(params.device)  
        idx = idx.unsqueeze(-1).to(params.device) 
        
        # Adversarial ground truths
        valid = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        
        labels = labels_batch[:,params.predict_start:]
        q50, q90 = model.forward(train_batch, idx)   
        d_loss = 0
        if params.gan=='False':
            optimizer_G.zero_grad()
            loss = transformer.loss_quantile(q50, labels, torch.tensor(0.5)) 
            loss.backward()
            optimizer_G.step()
            g_loss = loss.item() / params.train_window
            loss_epoch[i] = g_loss
        else:
            fake_input = torch.cat((labels_batch[:,:params.predict_start], q50), 1)
            #-------------------------------------------------------------------
            # Train the generator 
            #-------------------------------------------------------------------
            optimizer_G.zero_grad()
            loss =  transformer.loss_quantile(q50, labels, torch.tensor(0.5)) + 0.1 * adversarial_loss(discriminator(fake_input), valid)
            loss.backward()
            optimizer_G.step()
            g_loss = loss.item() / params.train_window
            loss_epoch[i] = g_loss

            #-------------------------------------------------------------------
            # Train the discriminator
            #-------------------------------------------------------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(labels_batch), valid)
            fake_loss = adversarial_loss(discriminator(fake_input.detach()), fake)
            loss_d = 0.5*(real_loss + fake_loss)
            loss_d.backward()
            optimizer_D.step()  
                
            d_loss = loss_d.item()
            d_loss_epoch[i] = d_loss
        
        if i % 1000 == 0:
            logger.info("G_loss: {} ; D_loss: {}".format(g_loss, d_loss))
            
    return loss_epoch, d_loss_epoch


def train_and_evaluate(model: nn.Module,
                       discriminator:nn.Module,
                       train_loader: DataLoader,
                       valid_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer_G,
                       optimizer_D,
                       adversarial_loss,
                       params: utils.Params,
                       restore_file: str = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    early_stopping = EarlyStopping(patience=100, verbose=True)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer_G)
    
    logger.info('begin training and evaluation')
    best_valid_q50 = float('inf')
    best_valid_q90 = float('inf')
    best_test_q50 = float('inf')
    best_test_q90 = float('inf')
    best_MAPE = float('inf')
    train_len = len(train_loader) 
    test_len = len(test_loader)

    q50_summary = np.zeros(params.num_epochs)
    q90_summary = np.zeros(params.num_epochs)
    MAPE_summary = np.zeros(params.num_epochs)

    q50_valid = np.zeros(params.num_epochs)
    q90_valid = np.zeros(params.num_epochs)
    MAPE_valid = np.zeros(params.num_epochs)

    loss_summary = np.zeros((train_len * params.num_epochs))
    loss_test = np.zeros((test_len * params.num_epochs))
    d_loss_summary = np.zeros((train_len * params.num_epochs))
    valid_loss = []
    logger.info("My Transformer have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for epoch in range(params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        
        loss_summary[epoch * train_len:(epoch + 1) * train_len], d_loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, discriminator,optimizer_G, optimizer_D, adversarial_loss, train_loader,
                                                                        valid_loader, params, epoch)
        
        test_metrics = evaluate(model, test_loader, params, epoch)
        valid_metrics = evaluate(model, valid_loader, params, epoch)
        loss_test[epoch * test_len:(epoch + 1) * test_len] = test_metrics['loss'].cpu()

        q50_valid[epoch] = valid_metrics['q50']
        q90_valid[epoch] = valid_metrics['q90']
        MAPE_valid[epoch] = valid_metrics['MAPE']

        q50_summary[epoch] = test_metrics['q50']
        q90_summary[epoch] = test_metrics['q90']
        MAPE_summary[epoch] = test_metrics['MAPE']

        valid_loss.append(valid_metrics['q50'])
        
        #is_best = q90_summary[epoch] <= best_test_q90
        is_best = q50_valid[epoch] <= best_valid_q50

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer_G.state_dict()},
                                epoch=epoch,
                                is_best=is_best,
                                checkpoint=params.model_dir)

        if is_best:
            logger.info('- Found new best Q90/Q50')
            best_valid_q50 = q50_summary[epoch]
            best_valid_q50 = q50_valid[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        utils.save_loss(loss_summary[epoch * train_len:(epoch + 1) * train_len], args.dataset + '_' + str(epoch) +'-th_epoch_loss', params.plot_dir)
        utils.save_loss(loss_test[epoch * test_len:(epoch + 1) * test_len], args.dataset + '_' + str(epoch) +'-th_epoch_test_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)
        early_stopping(valid_loss[-1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            # save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                filepath=params.model_dir)
            break

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = list(params.__dict__.keys())
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()

if __name__ == '__main__':
    one = torch.FloatTensor([1])
    mone = one * -1

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    log_file = os.path.join(model_dir, 'train.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    params.relative_metrics = args.relative_metrics
    params.attn_transform = args.attn_transform
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.dataset = args.dataset

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass
    
    # use GPU if available
    params.ngpu = torch.cuda.device_count()

    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info('Using Cuda...')
    c = copy.deepcopy
    attn = transformer.MultiHeadedAttention(params)
    ff = transformer.PositionwiseFeedForward(params.d_model, d_ff=params.d_ff, dropout=params.dropout)
    position = transformer.PositionalEncoding(params.d_model, dropout=params.dropout)
    #pt = transformer.TimeEncoding(params.d_model, dropout=0.1).cuda()
    ge = transformer.Generator(params)
    emb = transformer.Embedding(params, position)
    
    model = transformer.EncoderDecoder(params= params, emb = emb, encoder = transformer.Encoder(params, transformer.EncoderLayer(params, c(attn), c(ff), dropout=params.dropout)), decoder = transformer.Decoder(params, transformer.DecoderLayer(params, c(attn), c(attn), c(ff), dropout=params.dropout)), generator = ge)
    discriminator = transformer.Discriminator(params)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        discriminator = nn.DataParallel(discriminator)

    model.to(params.device)
    discriminator.to(params.device)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')

    train_set = TrainDataset(data_dir, args.dataset, params.num_class)
    valid_set = ValidDataset(data_dir, args.dataset, params.num_class)
    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    #sampler = WeightedSampler(data_dir, args.dataset) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=params.predict_batch, sampler=RandomSampler(valid_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    n_updates_total = (train_set.__len__() // params.batch_size) * params.num_epochs

    optimizer_D = optim.RMSprop(discriminator.parameters(), lr = params.lr_d)
    optimizer_G = OpenAIAdam(model.parameters(),
                           lr=params.lr,
                           schedule=params.lr_schedule,
                           warmup=params.lr_warmup,
                           t_total=n_updates_total,
                           b1=0.9,
                           b2=0.999,
                           e=1e-8,
                           l2=0.01,
                           vector_l2='store_true',
                           max_grad_norm=1)

    adversarial_loss = torch.nn.BCELoss()
    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,
                       discriminator,
                       train_loader,
                       valid_loader,
                       test_loader,
                       optimizer_G,
                       optimizer_D,
                       adversarial_loss,
                       params,
                       args.restore_file)
