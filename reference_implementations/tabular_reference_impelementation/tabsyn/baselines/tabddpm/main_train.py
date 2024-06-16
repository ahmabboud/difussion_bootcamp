import os
import argparse

from baselines.tabddpm.tabddpm import train

import src
import numpy as np
from tabsyn.utils import make_dataset


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = "/projects/aieng/diffussion_bootcamp/models/tabular"
    dataname = args.dataname
    device = f"cuda:{args.gpu}"

    config_path = f"{curr_dir}/configs/{dataname}.toml"
    model_save_path = f"{ckpt_dir}/tabddpm/{dataname}"
    real_data_path = f"/projects/aieng/diffussion_bootcamp/data/tabular/{dataname}"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    args.train = True
    raw_config = src.load_config(config_path)

    real_data_path = os.path.normpath(real_data_path)

    # zero.improve_reproducibility(seed)

    T = src.Transformations(**raw_config["train"]["T"])

    dataset = make_dataset(
        real_data_path,
        T,
        task_type=raw_config["task_type"],
        change_val=False,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or raw_config["train"]["T"]["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features = (
        dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    )
    d_in = np.sum(K) + num_numerical_features
    raw_config["model_params"]["d_in"] = d_in
    print(d_in)

    """ 
    Modification of configs
    """
    print("START TRAINING")

    train(
        **raw_config["train"]["main"],
        **raw_config["diffusion_params"],
        dataset=dataset,
        model_save_path=model_save_path,
        num_classes=K,
        model_type=raw_config["model_type"],
        model_params=raw_config["model_params"],
        T_dict=raw_config["train"]["T"],
        num_numerical_features=num_numerical_features,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", metavar="FILE")
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
