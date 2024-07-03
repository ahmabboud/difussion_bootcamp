import os
import torch
import json
import time

import numpy as np
from copy import deepcopy
import pandas as pd
import src
from src.baselines.tabddpm.model.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from src.baselines.tabddpm.model.denoise_model import (
    MLPDiffusion,
)
from src.baselines.tabddpm.utils import (
    split_num_cat_target,
    recover_data,
)


class Trainer:
    def __init__(
        self,
        diffusion,
        train_iter,
        lr,
        weight_decay,
        steps,
        model_save_path,
        device=torch.device("cuda:1"),
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.model_save_path = model_save_path

        columns = list(np.arange(5) * 200)
        columns[0] = 1
        columns = ["step"] + columns

        self.log_every = 50
        self.print_every = 1
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)

        self.optimizer.zero_grad()

        loss_multi, loss_gauss = self.diffusion.mixed_loss(x)

        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print("Steps: ", self.steps)
        while step < self.steps:
            start_time = time.time()
            x = next(self.train_iter)[0]

            batch_loss_multi, batch_loss_gauss = self._run_step(x)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if np.isnan(gloss):
                    print("Finding Nan")
                    break

                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}"
                    )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]

                np.set_printoptions(suppress=True)

                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

                if mloss + gloss < best_loss:
                    best_loss = mloss + gloss
                    torch.save(
                        self.diffusion._denoise_fn.state_dict(),
                        os.path.join(self.model_save_path, "model.pt"),
                    )

                if (step + 1) % 10000 == 0:
                    torch.save(
                        self.diffusion._denoise_fn.state_dict(),
                        os.path.join(self.model_save_path, f"model_{step+1}.pt"),
                    )

            # update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            end_time = time.time()
            print("Time: ", end_time - start_time)


class TabDDPM:
    def __init__(
        self,
        dataset,
        real_data_path,
        model_type,
        model_params,
        num_numerical_features,
        num_classes,
        num_timesteps=1000,
        seed=2024,
        gaussian_loss_type="mse",
        scheduler="cosine",
        change_val=False,
        device=torch.device("cuda:0"),
    ):
        self.seed = seed
        self.dataset = dataset
        self.num_numerical_features = num_numerical_features
        self.device = device
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.change_val = change_val
        self.info_path = f"{real_data_path}/info.json"
        print(model_params)
        self.model = self.get_model(
            model_type,
            model_params,
            self.num_numerical_features,
            category_sizes=self.dataset.get_category_sizes("train"),
        )
        self.model.to(device)
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device,
        )

        num_params = sum(p.numel() for p in self.diffusion.parameters())
        print("the number of parameters", num_params)

        self.diffusion.to(self.device)

    def get_model(self, model_name, model_params, n_num_features, category_sizes):
        print(model_name)
        if model_name == "mlp":
            model = MLPDiffusion(**model_params)
        else:
            raise "Unknown model!"
        return model

    def train(
        self,
        model_save_path,
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
    ):
        train_loader = src.prepare_fast_dataloader(
            self.dataset, split="train", batch_size=batch_size
        )

        trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            model_save_path=model_save_path,
            device=self.device,
        )
        trainer.run_loop()

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        torch.save(
            self.diffusion._denoise_fn.state_dict(),
            os.path.join(model_save_path, "model.pt"),
        )
        torch.save(
            trainer.ema_model.state_dict(),
            os.path.join(model_save_path, "model_ema.pt"),
        )

        trainer.loss_history.to_csv(
            os.path.join(model_save_path, "loss.csv"), index=False
        )

    def sample(
        self,
        sample_save_path,
        ckpt_path,
        batch_size=2000,
        num_samples=1000,
        disbalance=None,
        ddim=False,
        steps=1000,
    ):
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print("Loaded model from", ckpt_path)

        self.diffusion.eval()

        start_time = time.time()
        if not ddim:
            x_gen = self.diffusion.sample_all(num_samples, batch_size, ddim=False)
        else:
            x_gen = self.diffusion.sample_all(
                num_samples, batch_size, ddim=True, steps=steps
            )

        print("Shape", x_gen.shape)

        syn_data = x_gen
        num_inverse = self.dataset.num_transform.inverse_transform
        cat_inverse = self.dataset.cat_transform.inverse_transform

        with open(self.info_path, "r") as f:
            info = json.load(f)

        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        end_time = time.time()

        print("Sampling time:", end_time - start_time)

        save_path = sample_save_path
        syn_df.to_csv(save_path, index=False)
