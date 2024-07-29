import os
import src
import json
import pandas as pd
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from scripts.download_dataset import download_from_uci
from scripts.process_dataset import process_data

from src.data import preprocess, TabularDataset
from src.baselines.tabsyn.pipeline import TabSyn


NAME_URL_DICT_UCI = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
}

DATA_DIR = "/projects/aieng/diffusion_bootcamp/data/tabular_copy"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
SYNTH_DATA_DIR = os.path.join(DATA_DIR, "synthetic_data")
DATA_NAME = "default"

MODEL_PATH = "/projects/aieng/diffusion_bootcamp/models/tabular/tabsyn_copy"

# download data
download_from_uci(DATA_NAME, RAW_DATA_DIR, NAME_URL_DICT_UCI)

# process data
INFO_DIR = "data_info"
process_data(DATA_NAME, INFO_DIR, DATA_DIR)

# review json file and its contents
with open(f"{PROCESSED_DATA_DIR}/{DATA_NAME}/info.json", "r") as file:
    data_info = json.load(file)
pprint(data_info)


# load config
config_path = os.path.join("src/baselines/tabsyn/configs", f"{DATA_NAME}.toml")
raw_config = src.load_config(config_path)

pprint(raw_config)

# preprocess data
X_num, X_cat, categories, d_numerical = preprocess(os.path.join(PROCESSED_DATA_DIR, DATA_NAME),
                                                   transforms = raw_config["transforms"],
                                                   task_type = raw_config["task_type"])

# separate train and test data
X_train_num, X_test_num = X_num
X_train_cat, X_test_cat = X_cat

# convert to float tensor
X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

# create dataset module
train_data = TabularDataset(X_train_num.float(), X_train_cat)

# move test data to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_num = X_test_num.float().to(device)
X_test_cat = X_test_cat.to(device)

# create train dataloader
train_loader = DataLoader(
    train_data,
    batch_size = raw_config["train"]["vae"]["batch_size"],
    shuffle = True,
    num_workers = raw_config["train"]["vae"]["num_dataset_workers"],
)

tabsyn = TabSyn(train_loader,
                X_test_num, X_test_cat,
                num_numerical_features = d_numerical,
                num_classes = categories,
                device = device)

# instantiate VAE model for training
tabsyn.instantiate_vae(**raw_config["model_params"], optim_params = raw_config["train"]["optim"]["vae"])

# train vae
tabsyn.train_vae(**raw_config["loss_params"],
                 num_epochs = raw_config["train"]["vae"]["num_epochs"],
                 save_path = os.path.join(MODEL_PATH, DATA_NAME, "vae"))

# embed all inputs in the latent space
tabsyn.save_vae_embeddings(X_train_num, X_train_cat,
                           vae_ckpt_dir = os.path.join(MODEL_PATH, DATA_NAME, "vae"))

# load latent space embeddings
train_z, _ = tabsyn.load_vae_embeddings(os.path.join(MODEL_PATH, DATA_NAME, "vae"))  # train_z dim: B x in_dim

# normalize embeddings
mean, std = train_z.mean(0), train_z.std(0)
train_z = (train_z - mean) / std
latent_train_data = train_z

# create data loader
latent_train_loader = DataLoader(
    latent_train_data,
    batch_size = raw_config["train"]["diffusion"]["batch_size"],
    shuffle = True,
    num_workers = raw_config["train"]["diffusion"]["num_dataset_workers"],
)

# instantiate diffusion model for training
tabsyn.instantiate_diffusion(in_dim = train_z.shape[1], hid_dim = train_z.shape[1], optim_params = raw_config["train"]["optim"]["diffusion"])

# train diffusion model
tabsyn.train_diffusion(latent_train_loader,
                       num_epochs = raw_config["train"]["diffusion"]["num_epochs"],
                       ckpt_path = os.path.join(MODEL_PATH, DATA_NAME))

