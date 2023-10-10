# ------------------------------------------------------------------------------------
# Modified from minDALL-E (https://github.com/kakaobrain/minDALL-E)
# ------------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('train.py'))))
# from dalle.models import Rep_Dalle
# from data_loader.dataset import CustomDataModule
# from data_loader.dataloader import CustomDataLoader
from data_loader.utils import mk_dataframe, data_setting, check_unopen_img, rm_unopen_img
# from logger.logger import setup_callbacks

seed = 42
path_upstream = 'minDALL-E/1.3B'
config_file = '../configs/CALL-E.yaml'
config_downstream = config_file
result_path = '../tf_model'
data_dir = './img_data'
n_gpus = 1
train, val = None, None
cleaning = False

def main():
    """
    transfer learning minDALL-E with Pixabay Custom dataset for text2image
    """
    torch.cuda.empty_cache()
    # mk dataframe
    df = mk_dataframe(seed)
    
    if cleaning:
        check_unopen_img(df)
        rm_unopen_img(df)

    # mk Custom dataset setting : transform variable, simply splited train/valid dataset (valid size=0.2, shuffle=True)
    data_transforms, train, valid = data_setting(df)

    pl.seed_everything(seed)
    