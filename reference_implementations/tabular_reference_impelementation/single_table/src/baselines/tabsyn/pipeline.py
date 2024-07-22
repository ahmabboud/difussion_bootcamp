import numpy as np
import time
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.baselines.tabsyn.model.vae import Model_VAE, Encoder_model, Decoder_model
from src.baselines.tabsyn.model.modules import MLPDiffusion, Model
from src.baselines.tabsyn.model.utils import sample
from src.baselines.tabsyn.utils import recover_data, split_num_cat_target


class TabSyn:
    def __init__(
        self,
        train_loader,
        X_test_num,
        X_test_cat,
        device,
        num_numerical_features,
        num_classes,
    ):
        
        self.train_loader = train_loader
        self.X_test_num = X_test_num
        self.X_test_cat = X_test_cat
        self.device = device
        self.d_numerical = num_numerical_features
        self.categories = num_classes

    def get_vae_model(self, n_head, factor, num_layers, d_token):
        model = Model_VAE(num_layers, self.d_numerical, self.categories, d_token, n_head = n_head, factor = factor, bias = True)
        model = model.to(self.device)

        pre_encoder = Encoder_model(num_layers, self.d_numerical, self.categories, d_token, n_head = n_head, factor = factor).to(self.device)
        pre_decoder = Decoder_model(num_layers, self.d_numerical, self.categories, d_token, n_head = n_head, factor = factor).to(self.device)

        pre_encoder.eval()
        pre_decoder.eval()

        return model, pre_encoder, pre_decoder
    
    def load_optim(self, model, lr, weight_decay, factor, patience):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, verbose=True)

        return optimizer, scheduler

    def train_vae(self, model, pre_encoder, pre_decoder, optimizer, scheduler, max_beta, min_beta, lambd, num_epochs, model_save_path, encoder_save_path, decoder_save_path, device):
        current_lr = optimizer.param_groups[0]['lr']
        patience = 0
        best_train_loss = float('inf')

        # training loop
        beta = max_beta
        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0

            curr_count = 0

            for batch_num, batch_cat in pbar:
                model.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)

                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
            
                loss_mse, loss_ce, loss_kld, train_acc = self.compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl    += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count
            

            '''
                Evaluation
            '''
            model.eval()
            with torch.no_grad():
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(self.X_test_num, self.X_test_cat)

                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = self.compute_loss(self.X_test_num, self.X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")
                    
                train_loss = val_loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience += 1
                    if patience == 10:
                        if beta > min_beta:
                            beta = beta * lambd


            # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
            print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))

        end_time = time.time()
        print('Training time: {:.4f} mins'.format((end_time - start_time)/60))

        # load and save encoder and decoder states
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        print('Successfully load and save the model!')

        return model, pre_encoder, pre_decoder
            
    def compute_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim = -1)
            acc += (x_hat == X_cat[:,idx]).float().sum()
            total_num += x_hat.shape[0]
        
        ce_loss /= (idx + 1)
        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc
    
    def save_vae_embeddings(self, pre_encoder, X_train_num, X_train_cat, vae_ckpt_dir, device):
        # Saving latent embeddings
        with torch.no_grad():
            X_train_num = X_train_num.to(device)
            X_train_cat = X_train_cat.to(device)

            train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

            np.save(f"{vae_ckpt_dir}/train_z.npy", train_z)

            print("Successfully save pretrained embeddings in disk!")

    def load_vae_embeddings(self, vae_ckpt_dir):
        embedding_save_path = f"{vae_ckpt_dir}/train_z.npy"
        train_z = torch.tensor(np.load(embedding_save_path)).float()

        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
        
        train_z = train_z.view(B, in_dim)

        return train_z, token_dim
    
    def get_diffusion_model(self, in_dim, hid_dim, device):
        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        print(denoise_fn)

        num_params = sum(p.numel() for p in denoise_fn.parameters())
        print("the number of parameters", num_params)

        model = Model(denoise_fn = denoise_fn, hid_dim = hid_dim).to(device)
        return model
    
    def train_diffusion(self, model, train_loader, optimizer, scheduler, num_epochs, ckpt_path, device):
        model.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

            batch_loss = 0.0
            len_input = 0
            for batch in pbar:
                inputs = batch.float().to(device)
                loss = model(inputs)
            
                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(model.state_dict(), f"{ckpt_path}/model.pt")
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f"{ckpt_path}/model_{epoch}.pt")

        end_time = time.time()
        print('Time: ', end_time - start_time)

    # def load_model(self, model, pre_encoder, pre_decoder, ckpt_dir, model_name):
    #     encoder_save_path = f"{ckpt_dir}/vae/encoder.pt"
    #     decoder_save_path = f"{ckpt_dir}/vae/decoder.pt"
    #     model_save_path = f"{ckpt_dir}/{model_name}"

    #     pre_encoder.load_state_dict(torch.load(encoder_save_path))
    #     pre_decoder.load_state_dict(torch.load(decoder_save_path))
    #     model.load_state_dict(torch.load(model_save_path))

    #     print("Loaded model from", ckpt_dir)

    #     return model, pre_encoder, pre_decoder

    def load_model(self, in_dim, hid_dim, ckpt_dir, device, d_numerical, categories, n_head, factor, num_layers, d_token):
        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        model = Model(denoise_fn = denoise_fn, hid_dim = hid_dim).to(device)
        model.load_state_dict(torch.load(f'{ckpt_dir}/model.pt'))

        pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)
        decoder_save_path = f"{ckpt_dir}/vae/decoder.pt"
        pre_decoder.load_state_dict(torch.load(decoder_save_path))

        return model, pre_decoder

    def sample(self, model, train_z, info, num_inverse, cat_inverse, save_path, device):
        '''
            Generating samples    
        '''
        in_dim = train_z.shape[1] 
        mean = train_z.mean(0)

        start_time = time.time()

        num_samples = train_z.shape[0]
        sample_dim = in_dim

        x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
        x_next = x_next * 2 + mean.to(device)

        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device) 

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index = False)
        
        end_time = time.time()
        print('Time:', end_time - start_time)

        print('Saving sampled data to {}'.format(save_path))