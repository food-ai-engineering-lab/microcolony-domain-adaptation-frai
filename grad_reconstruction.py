import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from architecture import MicrocolonyNet
from datatools import McolonyTestData
from dann import DANN
from prototype_net import ProtoTypeDANN
# Input image dimensions (match this with your dataset)
max_px = 1002
min_px = 1002

def main(args):    
    # Arguments from command line
    root_train = args.root_train
    root = args.root
    workers = args.workers
    batch = args.batch
    ckpt = args.ckpt
    
    # Set multiprocessing strategy to 'file_system' for pytorch
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Get class names from the training dataset directory
    ds_classes = sorted(os.listdir(root_train))
    print("Classes: ", ds_classes)
    
    # Get experimental group name from test dataset directory
    ds_group = root.split('/')[-2]
    
    # Initialize DataModule   
    print("Loading dataset from ", root)
    ds = McolonyTestData(root=root)
    test_dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)

    model= DANN.load_from_checkpoint(ckpt)
    model.cuda()    # Move model to GPU
    for param in model.parameters():
        param.requires_grad = False
    print('Model loaded')

    itt = iter(test_dl)
    batches = int(np.ceil(len(ds)/batch))

    name_list = []
    pred_list = []
    gt_list = []

    # Loop over data batches
    for i in tqdm(range(batches)):
        d = next(itt)
        input = d[0]['image'].to('cuda')    # move input data to GPU
        fname = d[1]

        noise_data = torch.randn_like(input) * 0.1
        noise_data.requires_grad = True

        # Perform inference
        model.eval()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD([noise_data], lr=0.03)

        with torch.no_grad():
            _, logits, _ = model(noise_data, alpha=1)
            softmax_logits = F.softmax(logits, dim=1)
            labels = torch.zeros_like(softmax_logits)
            labels[:, 0] = 1
        for i in range(500):
            _, logits, _ = model(noise_data, alpha=1)
            softmax_logits = F.softmax(logits, dim=1)
            loss = criterion(softmax_logits, labels)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss.item())

            if loss.item() < 0.01:
                break
        

        
        
        exit(0)


if __name__ == '__main__':
    # Define and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-rt', '--root_train', type=str, help='Root folder of the training dataset', required=False,
        #default='/mnt/projects/bhatta70/train/'
        #default='/mnt/projects/bhatta70/train/20x/train90/'
        default='/mnt/data/mcolony-classification/20x/train90/'
        )
    parser.add_argument('-r', '--root', type=str, help='Root folder of the test dataset', required=False,
        default='/mnt/projects/bhatta70/test/brightfield/'
        # default = '/mnt/projects/bhatta70/test/20x/test15/'
        # default = '/mnt/projects/bhatta70/test/20x-5h/'
        # default='/mnt/projects/bhatta70/test/defocus/'
        # default='/mnt/projects/bhatta70/test/phase_contrast/'
        )
    parser.add_argument('-c', '--ckpt', type=str, help='Path to checkpoint file', default='/mnt/projects/bhatta70/Microcolony-AutoAugment/lightning_logs/mcolony/version_0/checkpoints/mcolony-epoch=31-val_loss_epoch=0.50.ckpt')
    parser.add_argument('-w', '--workers', type=int, help='Number of dataloader workers per GPU', default=0)
    parser.add_argument('-b', '--batch', type=int, help='Batch size per GPU', default=1)
    args = parser.parse_args()
    main(args)
