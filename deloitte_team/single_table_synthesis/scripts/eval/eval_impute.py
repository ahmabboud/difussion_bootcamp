import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from src import load_config


# def eval_impute(dataname, processed_data_dir, impute_path, col=0):

#     # set paths
#     data_dir = os.path.join(processed_data_dir, dataname)
#     real_path = f"{data_dir}/test.csv"

#     # get model config
#     config_path = os.path.join("src/baselines/tabsyn/configs", f"{dataname}.toml")
#     raw_config = load_config(config_path)
#     # number of resampling trials in imputation
#     num_trials = raw_config["impute"]["num_trials"]

#     encoder = OneHotEncoder()

#     real_data = pd.read_csv(real_path)
#     target_col = real_data.columns[-1]
#     real_target = real_data[target_col].to_numpy().reshape(-1, 1)
#     real_y = encoder.fit_transform(real_target).toarray()

#     syn_y = []
#     for i in range(num_trials):
#         syn_path = os.path.join(impute_path, dataname, f"{i}.csv")
#         syn_data = pd.read_csv(syn_path)
#         target = syn_data[target_col].to_numpy().reshape(-1, 1)
#         syn_y.append(encoder.transform(target).toarray())

#     syn_y = np.stack(syn_y).mean(0)

#     # Calculate metrics
#     real_labels = real_y.argmax(axis=1)
#     syn_labels = syn_y.argmax(axis=1)

#     micro_f1 = f1_score(real_labels, syn_labels, average="micro")
#     auc = roc_auc_score(real_y, syn_y, average="micro")

#     # Print metrics
#     print("Micro-F1:", micro_f1)
#     print("AUC:", auc)

#     # Calculate confusion matrix
#     cm = confusion_matrix(real_labels, syn_labels)

#     # Get the class labels from the encoder
#     class_labels = encoder.categories_[0]

#     # Convert confusion matrix to DataFrame for better readability with labeled axes
#     cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in class_labels],
#                          columns=[f"Predicted {label}" for label in class_labels])

#     # Print confusion matrix in a 'teddy' format with actual and predicted labels
#     print("\nConfusion Matrix:")
#     print(cm_df)





def eval_impute(dataname, processed_data_dir, impute_path, col=0):
    # set paths
    data_dir = os.path.join(processed_data_dir, dataname)
    real_path = f"{data_dir}/test.csv"

    # get model config
    config_path = os.path.join("src/baselines/tabsyn/configs", f"{dataname}.toml")
    raw_config = load_config(config_path)
    num_trials = raw_config["impute"]["num_trials"]

    encoder = OneHotEncoder()

    real_data = pd.read_csv(real_path)
    target_col = real_data.columns[-1]
    real_target = real_data[target_col].to_numpy().reshape(-1, 1)
    real_y = encoder.fit_transform(real_target).toarray()

    syn_y = []
    for i in range(num_trials):
        syn_path = os.path.join(impute_path, dataname, f"{i}.csv")
        syn_data = pd.read_csv(syn_path)
        target = syn_data[target_col].to_numpy().reshape(-1, 1)
        syn_y.append(encoder.transform(target).toarray())

    syn_y = np.stack(syn_y).mean(0)

    # Calculate real and synthetic labels
    real_labels = real_y.argmax(axis=1)
    syn_labels = syn_y.argmax(axis=1)

    # Calculate metrics
    micro_f1 = f1_score(real_labels, syn_labels, average="micro")
    auc = roc_auc_score(real_y, syn_y, average="micro")
    precision = precision_score(real_labels, syn_labels, average="micro")
    recall = recall_score(real_labels, syn_labels, average="micro")

    # Calculate confusion matrix
    cm = confusion_matrix(real_labels, syn_labels)
    
    # Get the class labels from the encoder
    class_labels = encoder.categories_[0]

    # Convert confusion matrix to DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in class_labels],
                         columns=[f"Predicted {label}" for label in class_labels])

    # Return the metrics and confusion matrix
    return {
        'Micro-F1': micro_f1,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': cm_df
    }