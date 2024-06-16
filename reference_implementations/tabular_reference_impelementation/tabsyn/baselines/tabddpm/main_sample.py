import os
import argparse
from baselines.tabddpm.tabddpm import sample

import src
from tabsyn.utils import make_dataset
import numpy as np


def main(args):
    dataname = args.dataname
    device = f"cuda:{args.gpu}"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = "/projects/aieng/diffussion_bootcamp/models/tabular"

    config_path = f"{curr_dir}/configs/{dataname}.toml"
    model_save_path = f"{ckpt_dir}/tabddpm/{dataname}"
    real_data_path = f"/projects/aieng/diffussion_bootcamp/data/tabular/{dataname}"
    sample_save_path = args.save_path

    args.train = True

    raw_config = src.load_config(config_path)

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
    raw_config["model_params"]["d_in"] = int(d_in)

    """ 
    Modification of configs
    """
    print("START SAMPLING")

    sample(
        dataset=dataset,
        num_samples=raw_config["sample"]["num_samples"],
        batch_size=raw_config["sample"]["batch_size"],
        disbalance=raw_config["sample"].get("disbalance", None),
        num_classes=K,
        **raw_config["diffusion_params"],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        model_type=raw_config["model_type"],
        model_params=raw_config["model_params"],
        num_numerical_features=num_numerical_features,
        device=device,
        ddim=args.ddim,
        steps=args.steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Whether to use ddim sampling.",
    )
    parser.add_argument("--steps", type=int, default=1000)

    args = parser.parse_args()
