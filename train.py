import os
import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch
from torch.multiprocessing import set_start_method
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datatools import McolonyDataModule
from architecture import MicrocolonyNet
from dann import DANN
from prototype_net import ProtoTypeDANN
def main(args):
    # Unpack the arguments from the command line
    root = args.root
    workers = args.workers
    batch = args.batch
    gpus = args.gpus
    accumulation = args.accumulation

    # Set multiprocessing start method to 'forkserver' for safer usage of multiprocessing
    set_start_method('forkserver')
    
    # Seed the random number generators for reproducible results
    pl.seed_everything(42, workers=True)     
               
    # Initialize DataModule, which loads and prepares the data
    dls = McolonyDataModule(root=root, dl_workers=workers, batch_size=batch)
    dls.setup()
    for (x,y) in dls.train_ds:
        print(x.shape)
        print(y)
        break
  
    
    # Initialize model
    # model = MicrocolonyNet()
    target_domains = ['bf']
    # target_domains = []
    model = DANN(target_domains=target_domains,
                    num_target_samples=3, mmd_weight=0, align_weight=1.5, cotrain_only=False)
    # model = ProtoTypeDANN(mmd_weight=1, align_weight=0, temperature=1)

    # Initialize checkpoint object to save model with best validation loss
    checkpoint = ModelCheckpoint(monitor='val_loss_epoch',
                                filename='mcolony-{epoch:02d}-{val_loss_epoch:.2f}',
                                save_top_k=10,
                                mode='min',
                                verbose=True)
    
    # Initialize LearningRateMonitor to log learning rate at each step
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize early stopping to halt training when validation loss stops improving
    # early_stopping = EarlyStopping(monitor='val_loss', patience=25
    #                             #    , check_finite=False
    #                                )
    
    # Initialize progress bar for visualizing training progress
    progress_bar = TQDMProgressBar(refresh_rate=25)
    
    logger = TensorBoardLogger('lightning_logs', name='mcolony-resnet50')
    
    num_epochs = 90
    # Initialize PyTorch Lightning Trainer with configurations
    trainer = pl.Trainer(num_nodes=gpus, max_epochs=num_epochs, callbacks=[checkpoint, lr_monitor, progress_bar],# early_stopping], 
                         profiler='simple', precision=16, deterministic=True,
                         logger = logger,
                         #detect_anomaly=True,
                        # accelerator='gpu'#, strategy='ddp'
                         )

    # Fit the model to the data
    trainer.fit(model, dls)
    
if __name__ == '__main__':
    # Set up argumetn parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, help='Root folder of the dataset', default='/mnt/projects/bhatta70/train/phase_contrast/')#required=True)
    parser.add_argument('-w', '--workers', type=int, help='Number of dataloader workers per GPU', default=4)
    parser.add_argument('-b', '--batch', type=int, help='Batch size per GPU', required=True)
    parser.add_argument('-g', '--gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('-a', '--accumulation', type=int, help='Number of accumulation steps', default=0)
    args = parser.parse_args()
    main(args)
