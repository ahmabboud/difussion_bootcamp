import os
import torch
import json
import time

import src
from baselines.tabddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from baselines.tabddpm.models.modules import MLPDiffusion
from baselines.tabddpm.train import Trainer
from baselines.tabddpm.sample import split_num_cat_target, recover_data


def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def train(
    dataset,
    model_save_path,
    num_classes,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False
):
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    print(model)

    train_loader = src.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=num_classes,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )

    num_params = sum(p.numel() for p in diffusion.parameters())
    print("the number of parameters", num_params)
    

    diffusion.to(device)

    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device
    )
    trainer.run_loop()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(model_save_path, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))

    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)

def sample(
    dataset,
    model_save_path,
    sample_save_path,
    real_data_path,
    num_classes,
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    change_val = False,
    ddim = False,
    steps = 1000,
):

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
   
    model_path =f'{model_save_path}/model.pt'

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )


    diffusion = GaussianMultinomialDiffusion(
        num_classes,
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps = steps)
    

    print('Shape', x_gen.shape)

    syn_data = x_gen
    num_inverse = dataset.num_transform.inverse_transform
    cat_inverse = dataset.cat_transform.inverse_transform
    
    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path
    syn_df.to_csv(save_path, index = False)