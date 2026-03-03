import os
import signal
import socket
import argparse

import time
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import SingleDeviceStrategy

from model.cst import ClfSegTransformer
from data.dataset import FSCSDatasetModule
from common.callbacks import CustomCheckpoint, OnlineLogger, CustomProgressBar


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Classification Segmentation Transformer for Few-Shot Classification and Segmentation')
    parser.add_argument('--datapath', type=str, default='~/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--backbone', type=str, default='vit-small', help='Backbone')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco'], help='Experiment benchmark')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--way', type=int, default=1, help='N-way for K-shot evaluation episode')
    parser.add_argument('--shot', type=int, default=1, help='K-shot for N-way K-shot evaluation episode: fixed to 1 for training')
    parser.add_argument('--batchsize', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--maxepochs', type=int, default=2000, help='Max iterations')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='4-fold validation fold')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--sup', type=str, default='mask', choices=['mask', 'pseudo'], help='Supervision level')
    parser.add_argument('--resume', action='store_true', help='Flag to resume a finished run')
    parser.add_argument('--vis', action='store_true', help='Flag to visualize. Use with --eval')
    parser.add_argument('--imgsize', type=int, default=800, help='image size')  # not variable
    parser.add_argument('--nlayers', type=int, default=12, help="n-layers to use for feature extraction")
    parser.add_argument('--use_text', type=bool, default=True, help='Use text correlation for pseudo-mask generation')
    parser.add_argument('--use_sam', type=bool, defaul=True, help='Use SAM for pseudo-mask generation')
    parser.add_argument('--debug', type=bool, action='store_true', help='Flag to set debug environment')

    args = parser.parse_args()

    if args.debug:
        args.nowandb = True

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Dataset initialization
    dm = FSCSDatasetModule(args)
    
    # Pytorch-lightning main trainer
    ckpt_callback = CustomCheckpoint(args)
    trainer = Trainer(strategy=DDPPlugin(find_unused_parameters=True), # DistributedDataParallel
                    #strategy=SingleDeviceStrategy(device=torch.device("cuda:2")), 
                    callbacks=[CustomCheckpoint(args), CustomProgressBar(args)],
                    #gradient_clip_val=1.0,
                    accelerator='gpu',
                    #devices=torch.cuda.device_count(),
                    devices=2,#[3] 
                    logger=False if args.nowandb or args.eval else OnlineLogger(args),
                    max_epochs=args.maxepochs,
                    num_sanity_val_steps=0,
                    #precision=16,
                    # Deprecated since version v1.6: weights_save_path has been deprecated in v1.6 and will be removed in v1.8
                    weights_save_path=ckpt_callback.modelpath,
                    # log_every_n_steps=1,  # default is 50
                    # profiler='advanced',  # this is cool!
                    )
    print('These are the detected devices', torch.cuda.device_count())
    if args.eval:
        # Loading the best model checkpoint from args.logpath
        modelpath = ckpt_callback.modelpath
        model = ClfSegTransformer.load_from_checkpoint(modelpath, args=args)
        trainer.test(model, dataloaders=dm.test_dataloader())
    else:
        # Train
        model = ClfSegTransformer(args)
        if os.path.exists(ckpt_callback.lastmodelpath):
            ckpt_path = ckpt_callback.lastmodelpath
            # PyTorch 1.12 ver issue; should assign capturable = True for rerun
            # https://github.com/pytorch/pytorch/issues/80831
            trainer.rerun = True
        else:
            ckpt_path = None
            trainer.rerun = False
        trainer.fit(model, dm, ckpt_path=ckpt_path)
