
import json
import argparse
import os
import sys
sys.path.append('/')
sys.path.append('../')
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, roc_curve
import numpy as np
import csv

file = 'eval_config.json'
with open(f'configs/{file}', 'r') as f:
    f_args = json.load(f)

parser = argparse.ArgumentParser(description=f_args["description"])
parser.add_argument("--workdir", type=str, default=f_args["workdir"])
parser.add_argument("--lambda_value", type=str, default=f_args["lambda_value"])
parser.add_argument("--metric", type=str, default=f_args["metric"])
parser.add_argument("--metric_inneed", type=list, default=f_args["metric_inneed"])
parser.add_argument("--metric2avg", type=list, default=f_args["metric2avg"])
args = parser.parse_args()
print(parser.description)

val_dir = f"{args.workdir}/valid/{args.lambda_value}/"
in_dir = f"{args.workdir}/test_in/{args.lambda_value}/"
ood_dir = f"{args.workdir}/test_out/{args.lambda_value}/"


def get_val_data(valid_dir, metric_inneed, loss_fn="loss.csv"):
    loss_path = os.path.join(valid_dir, loss_fn)
    df_all = pd.read_csv(loss_path)
    return_data1, return_data2 = {}, {}
    for key in metric_inneed:
        data_list = list(df_all[key])
        return_data1[key] = (np.mean(data_list), np.std(data_list))
        return_data2[key] = np.array(data_list)

    return list(df_all["file_name"]), return_data1, return_data2

def get_in_data(in_dir, metric_inneed, loss_fn="loss.csv"):
    loss_path = os.path.join(in_dir, loss_fn)
    df_all = pd.read_csv(loss_path)
    df_all = df_all.drop_duplicates(subset=["file_name"])
    return_data = {}
    for key in metric_inneed:
        data_list = list(df_all[key])
        return_data[key] = np.array(data_list)

    return list(df_all["file_name"]), return_data

def get_ood_data(ood_dir, metric_inneed, loss_fn="loss.csv"):
    loss_path = os.path.join(ood_dir, loss_fn)
    df_all = pd.read_csv(loss_path)
    df_all = df_all.drop_duplicates(subset=["file_name"])
    df_all = df_all.loc[~df_all["file_name"].isin(filter_out_list)]
    label = [fn[-7: -4] for fn in df_all["file_name"]]
    df_all["label"] = label
    df_all_weak = df_all.loc[(df_all["label"] != "000") & (df_all["label"] != "200")]
    df_all_strong = df_all.loc[(df_all["label"] == "100") | (df_all["label"] == "111")]
    df_all_normal = df_all.loc[df_all["label"] == "000"]
    return_data = {}
    for key in metric_inneed:
        return_data[key] = (np.array(list(df_all_normal[key])), np.array(list(df_all_weak[key])), np.array(list(df_all_strong[key])), np.array(list(df_all[key])))

    return list(df_all["file_name"]), return_data

if __name__ == "__main__":

    val_fn, val_mean_std, val_data = get_val_data(val_dir, args.metric_inneed)
    in_fn, in_data = get_in_data(in_dir, args.metric_inneed)
    ood_filename, ood_data = get_ood_data(ood_dir, args.metric_inneed)

    with open(os.path.join(val_dir, "z_score.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file_name"] + args.metric_inneed)
        writer.writeheader()
    val_z_score_df = pd.read_csv(os.path.join(val_dir, "z_score.csv"))
    val_z_score_df["file_name"] = val_fn

    with open(os.path.join(in_dir, "z_score.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file_name"] + args.metric_inneed)
        writer.writeheader()
    in_z_score_df = pd.read_csv(os.path.join(in_dir, "z_score.csv"))
    in_z_score_df["file_name"] = in_fn

    with open(os.path.join(ood_dir, "z_score.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file_name"] + args.metric_inneed)
        writer.writeheader()
    ood_z_score_df = pd.read_csv(os.path.join(ood_dir, "z_score.csv"))
    ood_z_score_df["file_name"] = ood_filename

    print(
        f"Lambda value: {args.lambda_value}: validation items: {len(val_fn)}, in item: {len(in_fn)}, ood items: {len(ood_filename)}")
    val_z_score, in_z_score, ood_z_score = {}, {}, {}
    for key in args.metric_inneed:
        val_z_score[key], in_z_score[key], ood_z_score[key] = {}, {}, {}
        val_z_score[key] = (val_data[key] - val_mean_std[key][0]) / val_mean_std[key][1]
        val_z_score_df[key] = val_z_score[key]

        in_z_score[key] = (in_data[key] - val_mean_std[key][0]) / val_mean_std[key][1]
        in_z_score_df[key] = in_z_score[key]

        ood_z_score[key]["in"] = (ood_data[key][0] - val_mean_std[key][0]) / val_mean_std[key][1]
        ood_z_score[key]["ood_full"] = (ood_data[key][2] - val_mean_std[key][0]) / val_mean_std[key][1]
        ood_z_score[key]["all"] = (ood_data[key][3] - val_mean_std[key][0]) / val_mean_std[key][1]
        ood_z_score_df[key] = ood_z_score[key]["all"]

    in_z_score["avg"] = np.zeros_like(in_z_score[args.metric2avg[0]])
    ood_z_score["avg"] = {
        "in": np.zeros_like(ood_z_score[args.metric2avg[0]]["in"]),
        "ood_full": np.zeros_like(ood_z_score[args.metric2avg[0]]["ood_full"]),
        "all": np.zeros_like(ood_z_score[args.metric2avg[0]]["all"])
    }
    for key in args.metric2avg:
        in_z_score["avg"] += in_z_score[key]
        ood_z_score["avg"]["in"] += ood_z_score[key]["in"]
        ood_z_score["avg"]["ood_full"] += ood_z_score[key]["ood_full"]

        in_z_score["avg"] += in_z_score[key]
        ood_z_score["avg"]["in"] += ood_z_score[key]["in"]
        ood_z_score["avg"]["ood_full"] += ood_z_score[key]["ood_full"]
        ood_z_score["avg"]["all"] += ood_z_score[key]["all"]

    in_z_score["avg"] /= len(args.metric2avg)
    ood_z_score["avg"]["in"] /= len(args.metric2avg)
    ood_z_score["avg"]["ood_full"] /= len(args.metric2avg)
    ood_z_score["avg"]["all"] /= len(args.metric2avg)

    val_z_score_df.to_csv(os.path.join(val_dir, "z_score.csv"), index=False)
    in_z_score_df.to_csv(os.path.join(in_dir, "z_score.csv"), index=False)
    ood_z_score_df.to_csv(os.path.join(ood_dir, "z_score.csv"), index=False)

    metric_key = ["ood_full"]
    for mc in metric_key:

        z_scores_combined = np.concatenate(
            [in_z_score[args.metric], ood_z_score[args.metric]['in'], ood_z_score[args.metric][mc]])
        labels = np.concatenate([np.zeros(len(in_z_score[args.metric]) + len(ood_z_score[args.metric]['in'])),
                                 np.ones(len(ood_z_score[args.metric][mc]))])
        z_scores_combined = np.concatenate(
            [in_z_score[args.metric], ood_z_score[args.metric][mc]])
        labels = np.concatenate([np.zeros(len(in_z_score[args.metric])),
                                 np.ones(len(ood_z_score[args.metric][mc]))])

        auc = roc_auc_score(labels, z_scores_combined)
        auc = round(auc, 4)
        print(f"Avg {mc} Original AUC: ", auc)
        aupr = average_precision_score(labels, z_scores_combined)
        aupr = round(aupr, 4)
        print(f"Avg {mc} Original AUPR: ", aupr)
