import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score


def main(dataname="default", col=0):
    data_dir = f"data/tabular/processed_data/{dataname}"

    real_path = f"{data_dir}/test.csv"


    encoder = OneHotEncoder()

    real_data = pd.read_csv(real_path)
    target_col = real_data.columns[-1]
    real_target = real_data[target_col].to_numpy().reshape(-1, 1)
    real_y = encoder.fit_transform(real_target).toarray()


    syn_y = []
    for i in range(9):
        syn_path = f"impute/tabsyn/{dataname}/{i}.csv"
        syn_data = pd.read_csv(syn_path)
        target = syn_data[target_col].to_numpy().reshape(-1, 1)
        syn_y.append(encoder.transform(target).toarray())

    syn_y = np.stack(syn_y).mean(0)


    micro_f1 = f1_score(real_y.argmax(axis=1), syn_y.argmax(axis=1), average="micro")
    auc = roc_auc_score(real_y, syn_y, average="micro")

    print("Micro-F1:", micro_f1)
    print("AUC:", auc)
