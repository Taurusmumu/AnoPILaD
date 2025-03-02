
import json
import argparse
import os
import sys
sys.path.append('/')
sys.path.append('../')
import pandas as pd
from scipy.ndimage import grey_erosion
import openslide
from xml.dom import minidom
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from sklearn.utils import resample
import cv2


file = 'eval_config.json'
with open(f'configs/{file}', 'r') as f:
    f_args = json.load(f)

parser = argparse.ArgumentParser(description=f_args["description"])
parser.add_argument("--workdir", type=str, default=f_args["workdir"])
parser.add_argument("--lambda_value", type=str, default=f_args["lambda_value"])
parser.add_argument("--nine_percent", type=float, default=f_args["nine_percent"])
parser.add_argument("--metric2avg", type=list, default=f_args["metric2avg"])
parser.add_argument("--erosion", type=list, default=f_args["erosion"])
args = parser.parse_args()
print(parser.description)

in_wsi_root_path = f_args["test_in_wsi"]
ood_wsi_root_path = f_args["test_out_wsi"]
ood_xml_root_path = f_args["test_out_label"]

in_rec_path = f"{args.workdir}/test_in/{args.lambda_value}"
ood_rec_path = f"{args.workdir}/test_out/{args.lambda_value}"

in_score_csv = os.path.join(in_rec_path, "z_score.csv")
pd_in = pd.read_csv(in_score_csv)
wsi_in_list = [fn.split('_')[0] for fn in list(pd_in["file_name"])]
pd_in["wsi"] = wsi_in_list
available_in_wsi = list(set(wsi_in_list))

ood_score_csv = os.path.join(ood_rec_path, "z_score.csv")
pd_ood = pd.read_csv(ood_score_csv)
wsi_ood_list = [fn.split('_')[0] for fn in list(pd_ood["file_name"])]
pd_ood["wsi"] = wsi_ood_list
available_ood_wsi = list(set(wsi_ood_list))

plot_pd_dict = {}
extract_layer = 0
rescale_rate = 256
patch_size = 256
output_patch_size = patch_size // rescale_rate
seed = 42
log_path = os.path.join(args.workdir, "heatmaps", args.lambda_value)


def vis_in_wsi_zscore_heatmap(wsi_path_base, pd_in_wsi):
    print(wsi_path_base)
    wsi_path = os.path.join(in_wsi_root_path, f'{wsi_path_base}.svs')
    # Load and downsample the WSI
    wsi_obj = openslide.OpenSlide(wsi_path)
    wsi_width, wsi_height = wsi_obj.level_dimensions[0]
    down_wsi_w, down_wsi_h = wsi_width / rescale_rate, wsi_height / rescale_rate# Full WSI dimensions
    thumbnail = wsi_obj.get_thumbnail([down_wsi_w, down_wsi_h]).convert("RGB")

    coordinates = [(int(fn.split('_')[1]), int(fn.split('_')[2])) for fn in list(pd_in_wsi["file_name"])]
    z_scores = list(pd_in_wsi["avg"])
    # Set up figure and overlay Z-score patches
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(thumbnail)
    cmap = plt.cm.YlOrRd  # Color map for z-score

    # Overlay each patch on the WSI
    for (x, y), z_score in zip(coordinates, z_scores):
        # Downsample coordinates to match the display resolution
        x_rescaled = x // rescale_rate
        y_rescaled = y // rescale_rate
        facecolor = cmap(z_score)
        rect = patches.Rectangle((x_rescaled, y_rescaled), output_patch_size, output_patch_size,
                                 linewidth=0, edgecolor=None, facecolor=facecolor, alpha=1)
        ax.add_patch(rect)

    # Add color bar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    # cbar.set_label("Z-score values", fontsize=12)

    # Optional: Customize labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(wsi_path_base, fontsize=16)
    plt.tight_layout()

    os.makedirs(log_path, exist_ok=True)
    fig.savefig(os.path.join(log_path, f"{wsi_path_base}_heatmap.png"))


def vis_ood_wsi_zscore_heatmap(wsi_path_base, pd_ood_wsi):
    print(wsi_path_base)
    wsi_path = os.path.join(ood_wsi_root_path, f'{wsi_path_base}.svs')
    # Load and downsample the WSI
    wsi_obj = openslide.OpenSlide(wsi_path)
    wsi_width, wsi_height = wsi_obj.level_dimensions[0]
    down_wsi_w, down_wsi_h = wsi_width / rescale_rate, wsi_height / rescale_rate # Full WSI dimensions
    thumbnail = wsi_obj.get_thumbnail([down_wsi_w, down_wsi_h]).convert("RGB")

    coordinates = [(int(fn.split('_')[1]), int(fn.split('_')[2])) for fn in list(pd_ood_wsi["file_name"])]
    z_scores = list(pd_ood_wsi["avg"])
    z_scores_oris = list(pd_ood_wsi["avg_original"])
    # Set up figure and overlay Z-score patches
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(thumbnail)
    cmap = plt.cm.YlOrRd  # Color map for z-score
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Overlay each patch on the WSI
    for (x, y), z_score, z_scores_ori in zip(coordinates, z_scores, z_scores_oris):
        # Downsample coordinates to match the display resolution
        x_rescaled = x // rescale_rate
        y_rescaled = y // rescale_rate
        facecolor = cmap(z_score)

        rect = patches.Rectangle((x_rescaled, y_rescaled), output_patch_size, output_patch_size,
                                 linewidth=0, edgecolor=None, facecolor=facecolor, alpha=1)
        ax.add_patch(rect)

    xml_path = os.path.join(ood_xml_root_path, f"{wsi_path_base}.xml")  # Replace with the path to the XML file
    contours_meta, contours_normal = parse_xml_contours(xml_path, rescale_rate)
    for contour in contours_meta:
        polygon = patches.Polygon(contour, closed=True, edgecolor='black', facecolor='none', linewidth=3)
        ax.add_patch(polygon)
    for contour in contours_normal:
        polygon = patches.Polygon(contour, closed=True, edgecolor='green', facecolor='none', linewidth=3)
        ax.add_patch(polygon)

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label("Z-score values", fontsize=12)

    # Optional: Customize labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(wsi_path_base, fontsize=16)
    plt.tight_layout()

    # Show the plot
    # plt.show()
    os.makedirs(log_path, exist_ok=True)
    fig.savefig(os.path.join(log_path, f"{wsi_path_base}_heatmap.png"))

def parse_xml_contours(xml_path, rescale_rate):

    def _createContour(coord_list):
        return np.array([[int(float(coord.attributes['X'].value) / rescale_rate),
                           int(float(coord.attributes['Y'].value) / rescale_rate)] for coord in coord_list], dtype='int32')

    xmldoc = minidom.parse(xml_path)
    tumor_coor_list, normal_coor_list = [], []
    annos = [anno for anno in xmldoc.getElementsByTagName('Annotation')]

    for anno in annos:
        if anno.getAttribute('Name') == 'LN_malignant':
            tumor_coor_list.append(anno.getElementsByTagName('Coordinate'))
        else:
            normal_coor_list.append(anno.getElementsByTagName('Coordinate'))
    contours_tumor = [_createContour(coord_list) for coord_list in tumor_coor_list]
    contours_tumor = sorted(contours_tumor, key=cv2.contourArea, reverse=True)
    contours_normal = [_createContour(coord_list) for coord_list in normal_coor_list]
    contours_normal = sorted(contours_normal, key=cv2.contourArea, reverse=True)
    return contours_tumor, contours_normal

def get_auc_aupr(wsi_labels, scores, method_name):
    auc = roc_auc_score(wsi_labels, scores)
    auc = round(auc, 4)
    n_samples = int(len(wsi_labels) * 0.8)
    n_bootstraps = 2000
    bootstrapped_aucs = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        if len(np.unique(wsi_labels[indices])) < 2:
            # Skip iteration if resample doesn't contain both classes
            continue
        auc_bs = roc_auc_score(wsi_labels[indices], scores[indices])
        bootstrapped_aucs.append(auc_bs)
    # Calculate the 95% confidence interval
    lower_bound = np.percentile(bootstrapped_aucs, 2.5)
    upper_bound = np.percentile(bootstrapped_aucs, 97.5)
    print(f"Mean of bootstrap AUC is {auc} [{round(lower_bound, 4)}, {round(upper_bound, 4)}]")

    aupr = average_precision_score(wsi_labels, scores)
    aupr = round (aupr, 4)

    bootstrapped_aupr = []
    for i in range(n_bootstraps):
        # Resample the data with replacement
        # indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples, random_state=seeds[i])
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        if len(np.unique(wsi_labels[indices])) < 2:
            # Skip iteration if resample doesn't contain both classes
            continue
        pr_auc = average_precision_score(wsi_labels[indices], scores[indices])
        bootstrapped_aupr.append(pr_auc)
    # Calculate the 95% confidence interval
    lower_bound = np.percentile(bootstrapped_aupr, 2.5)
    upper_bound = np.percentile(bootstrapped_aupr, 97.5)
    print(
        f"Mean of bootstrap AUPR is {aupr} [{round(lower_bound, 4)}, {round(upper_bound, 4)}]")


def erosion(wsi_root_path, wsi_path_base, patchs_df):
    print(wsi_path_base)
    wsi_path = os.path.join(wsi_root_path, f'{wsi_path_base}.svs')
    # Load and downsample the WSI
    wsi_obj = openslide.OpenSlide(wsi_path)
    wsi_width, wsi_height = wsi_obj.level_dimensions[0]
    down_wsi_w, down_wsi_h = wsi_width / rescale_rate, wsi_height / rescale_rate  # Full WSI dimensions
    canvas = np.zeros((int(down_wsi_h), int(down_wsi_w))).astype(np.float64)
    canvas += np.inf
    coordinates = [(int(fn.split('_')[1]), int(fn.split('_')[2])) for fn in list(patchs_df["file_name"])]

    for key in ["avg", "avg_original"]:
        z_scores = list(patchs_df[key])
        for (x, y), z_score in zip(coordinates, z_scores):
            # Downsample coordinates to match the display resolution
            x_rescaled = x // rescale_rate
            y_rescaled = y // rescale_rate
            canvas[y_rescaled][x_rescaled] = z_score
        ero_canvas = grey_erosion(canvas, size=(2,2))
        ero_z_scores = []
        for (x, y), z_score in zip(coordinates, z_scores):
            # Downsample coordinates to match the display resolution
            x_rescaled = x // rescale_rate
            y_rescaled = y // rescale_rate
            ero_z_scores.append(ero_canvas[y_rescaled][x_rescaled])
        patchs_df[key] = ero_z_scores

    return patchs_df


if __name__ == "__main__":
    in_len, out_len = len(pd_in["file_name"]), len(pd_ood["file_name"])
    all_z_score = np.zeros(in_len + out_len)
    for metric_key in args.metric2avg:
        all_z_score += np.array(list(pd_in[metric_key]) + list(pd_ood[metric_key]))

    all_z_score /= len(args.metric2avg)
    sorted_indices = np.argsort(all_z_score)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(all_z_score))

    # Calculate percentile ranks (scaled to 0-1)
    all_z_score_ = ranks / (len(all_z_score) - 1)
    pd_in['avg'] = all_z_score_[:in_len]
    pd_ood['avg'] = all_z_score_[in_len:]
    pd_in['avg_original'] = all_z_score[:in_len]
    pd_ood['avg_original'] = all_z_score[in_len:]

    wsi_aggregated_z_scores_max = []
    wsi_aggregated_z_scores_99 = []
    wsi_labels = []
    wsi_names = []
    re_fpr_all = []
    dice_all = []
    iou_all = []


    in_99_fn_names, out_99_fn_names = {}, {}
    for wsi_path_base in available_in_wsi:
        wsi_labels.append(0)
        wsi_names.append(wsi_path_base)
        patchs_df = pd_in.loc[pd_in["wsi"] == wsi_path_base]
        if args.erosion:
            patchs_df = erosion(in_wsi_root_path, wsi_path_base, patchs_df)
        plot_pd_dict[wsi_path_base] = patchs_df
        patch_fn = list(patchs_df['file_name'])
        patch_z_scores = np.array(list(patchs_df['avg']))

        wsi_aggregated_z_scores_max.append(max(patch_z_scores))
        sort_idx = np.argsort(patch_z_scores)
        patch_z_scores_99 = np.array(patch_z_scores)[sort_idx][int(len(patch_z_scores) * args.nine_percent):]
        patch_fn_99 = np.array(patch_fn)[sort_idx][int(len(patch_z_scores) * args.nine_percent):]
        in_99_fn_names[wsi_path_base] = list(patch_fn_99)
        # patch_z_scores_99 = sorted(patch_z_scores)[int(len(patch_z_scores) * args.nine_percent):]
        wsi_aggregated_z_scores_99.append(sum(patch_z_scores_99) / len(patch_z_scores_99))

        in_prediction = np.zeros_like(patch_z_scores)
        in_prediction[patch_z_scores > 0] = 1
        in_fp = np.sum(in_prediction)
        in_tn = len(in_prediction) - in_fp
        re_fpr_all.append(1 - in_fp / (in_fp + in_tn + 1e-8))

    for wsi_path_base in available_ood_wsi:
        wsi_labels.append(1)
        wsi_names.append(wsi_path_base)
        patchs_df = pd_ood.loc[pd_ood["wsi"] == wsi_path_base]
        if args.erosion:
            patchs_df = erosion(ood_wsi_root_path, wsi_path_base, patchs_df)
        plot_pd_dict[wsi_path_base] = patchs_df
        patch_fn = list(patchs_df['file_name'])
        patch_z_scores = np.array(list(patchs_df['avg']))
        patch_z_scores_ori = np.array(list(patchs_df['avg_original']))

        wsi_aggregated_z_scores_max.append(max(patch_z_scores))
        sort_idx = np.argsort(patch_z_scores)
        patch_z_scores_99 = np.array(patch_z_scores)[sort_idx][int(len(patch_z_scores) * args.nine_percent):]
        patch_fn_99 = np.array(patch_fn)[sort_idx][int(len(patch_z_scores) * args.nine_percent):]
        out_99_fn_names[wsi_path_base] = list(patch_fn_99)

        # patch_z_scores = list(pd_ood.loc[pd_ood["wsi"] == wsi_path_base]['avg'])
        # wsi_aggregated_z_scores_max.append(max(patch_z_scores))
        # patch_z_scores_99 = sorted(patch_z_scores)[int(len(patch_z_scores) * args.nine_percent):]
        wsi_aggregated_z_scores_99.append(sum(patch_z_scores_99) / len(patch_z_scores_99))

        ood_prediction = np.zeros_like(patch_z_scores_ori)
        ood_prediction[patch_z_scores_ori > 0] = 1
        ood_label = np.array([1 if ((fn[-7: -4] != '200') & (fn[-7: -4] != '000')) else 0 for fn in patch_fn])
        ood_tp = np.sum((ood_label == 1) & (ood_prediction == 1))
        ood_fp = np.sum((ood_label == 0) & (ood_prediction == 1))
        ood_fn = np.sum((ood_label == 1) & (ood_prediction == 0))
        dice = 2 * ood_tp / (2 * ood_tp + ood_fp + ood_fn + 1e-8)
        iou = ood_tp / (ood_tp + ood_fp + ood_fn + 1e-8)

        dice_all.append(dice)
        iou_all.append(iou)

    wsi_aggregated_z_scores_max = np.array(wsi_aggregated_z_scores_max)
    wsi_aggregated_z_scores_99 = np.array(wsi_aggregated_z_scores_99)
    wsi_all = np.array(available_in_wsi + available_ood_wsi)
    wsi_labels = np.array(wsi_labels)
    threshold = np.median(wsi_aggregated_z_scores_99)
    prediction = np.zeros_like(wsi_labels)
    prediction[np.array(wsi_aggregated_z_scores_99) >= threshold] = 1

    get_auc_aupr(wsi_labels, wsi_aggregated_z_scores_max, "MAX Z_score")
    get_auc_aupr(wsi_labels, wsi_aggregated_z_scores_99, "Average 99 Percentage Z_score")

    # for wsi_path_base in available_in_wsi:
    #     vis_in_wsi_zscore_heatmap(wsi_path_base, plot_pd_dict[wsi_path_base])
    for wsi_path_base in available_ood_wsi:
        vis_ood_wsi_zscore_heatmap(wsi_path_base, plot_pd_dict[wsi_path_base])



