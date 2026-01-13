import sys
import os
import re
import pandas as pd
import numpy as np
from sklearn.cluster import k_means
import matplotlib.pyplot as plt

exp_arg_count = 3
if len(sys.argv) != exp_arg_count + 1:
    print("Error: Expected {} argument: fsa_folder_path, annotations_folder_path, output_folder_path".format(exp_arg_count))
    sys.exit()
fsa_folder_path = sys.argv[1]
annotations_folder_path = sys.argv[2]
output_folder_path = sys.argv[3]

#######################################################################################################################
# Get plant names
def get_plant_names(fsa_folder_path):
    fsa_filenames = []
    pattern = re.compile(r'.*\.fsa')
    for filename in os.listdir(fsa_folder_path):
        if pattern.match(filename):
            fsa_filenames.append(filename)
    fsa_filename_df = pd.DataFrame(fsa_filenames, columns=["name"])
    fsa_filename_df.sort_values(by="name", inplace=True)
    fsa_filename_df.reset_index(drop=True, inplace=True)
    return fsa_filename_df

#######################################################################################################################
# Get labels and merge with plant names + remove plants containing a NA
def link_plant_names(annotations_folder_path, fsa_filename_df):
    anno_filenames = []
    pattern = re.compile(r'channel_.*\.csv')
    for filename in os.listdir(annotations_folder_path):
        if pattern.match(filename):
            anno_filenames.append(filename)
    all_anno_df_with_na = []
    df_na_idx_concat = []
    for anno_filename in anno_filenames:
        anno_df_with_na = pd.read_csv(annotations_folder_path+"/"+anno_filename, sep=";", na_values="NA")
        anno_df_with_na["channel"] = anno_filename[0:-4] # remove .csv
        all_anno_df_with_na.append(anno_df_with_na)
        df_na_idx_A1 = anno_df_with_na['markA.1'].index[anno_df_with_na['markA.1'].apply(np.isnan)]
        df_na_idx_A2 = anno_df_with_na['markA.2'].index[anno_df_with_na['markA.2'].apply(np.isnan)]
        df_na_idx_concat = np.concatenate([df_na_idx_concat, df_na_idx_A1, df_na_idx_A2])
    unique_na_idx = np.unique(df_na_idx_concat)
    fsa_filename_df_no_na = fsa_filename_df.drop(unique_na_idx, axis=0)
    all_anno_df_with_plant_names = []
    for anno_df_with_na in all_anno_df_with_na:
        anno_df = anno_df_with_na.drop(unique_na_idx, axis=0)
        anno_df = pd.concat([anno_df, fsa_filename_df_no_na], axis=1)
        anno_df.reset_index(drop=True, inplace=True)
        anno_df.rename(columns={"plant": "plant_idx", "name": "plant_name", "markA.1": "A1_weight_raw", "markA.2": "A2_weight_raw"}, inplace=True)
        anno_df = anno_df[["plant_idx", "plant_name", "channel", "A1_weight_raw", "A2_weight_raw"]]
        all_anno_df_with_plant_names.append(anno_df)
    return all_anno_df_with_plant_names

#######################################################################################################################
# Identify allele clusters and group them
def check_good_cluster_count(cluster_count, X, median_dist_thres):
    centroid, label, inertia = k_means(X.reshape(-1, 1), n_clusters=cluster_count, random_state=0, n_init="auto")
    centers = np.array(sorted(centroid[:, 0]))
    distances = []
    for i in range(len(centers) - 1):
        distances.append(centers[i + 1] - centers[i])
    median_dist = np.median(distances)
    #print(median_dist, distances)
    for i in range(len(centers) - 1):
        if centers[i + 1] - centers[i] < median_dist * median_dist_thres:
            return False
    return centroid[:, 0], label, inertia

def try_clusters(X, median_dist_thres):
    max_cluster_count = len(np.unique(X))
    for cluster_count in range(max_cluster_count, 0, -1):
        res = check_good_cluster_count(cluster_count, X, median_dist_thres)
        if res != False:
            centroid, labels, inertia = res
            # Sort centroids
            sorted_centroid = np.array(sorted(centroid))
            corr = {}
            for i in range(len(centroid)):
                corr[i] = np.where(sorted_centroid == centroid[i])[0][0]
            sorted_labels = []
            for l in labels:
                sorted_labels.append(corr[l])
            return sorted_centroid, sorted_labels, inertia
    print("Best cluster count identification failed")
    sys.exit()

def group_alleles(all_anno_df):
    median_dist_thres = 0.3
    for anno_df in all_anno_df:
        A1_weight = np.array(anno_df['A1_weight_raw'].to_list())
        centroid, labels, inertia = try_clusters(A1_weight, median_dist_thres)
        anno_df["A1_idx"] = labels
        anno_df["A1_weight"] = centroid[labels].astype(np.int64)
        A2_weight = np.array(anno_df['A2_weight_raw'].to_list())
        centroid, labels, inertia = try_clusters(A2_weight, median_dist_thres)
        anno_df["A2_idx"] = labels
        anno_df["A2_weight"] = centroid[labels].astype(np.int64)

#######################################################################################################################
# Function calls
fsa_filename_df = get_plant_names(fsa_folder_path)
# print(fsa_filename_df)
all_anno_df = link_plant_names(annotations_folder_path, fsa_filename_df)
# print(all_anno_df)
group_alleles(all_anno_df)
print(all_anno_df)

#######################################################################################################################
# Scatter plot to visualize allele group relevance
for anno_df in all_anno_df:
    for allele in range(1, 3):
        channel = anno_df["channel"].iloc[0]
        A1_raw = np.array(anno_df["A{}_weight_raw".format(allele)].to_list())
        A1_grouped = np.array(anno_df["A{}_weight".format(allele)].to_list())
        weights = np.unique(A1_grouped)
        A1_w = []
        for w in weights:
            A1_w.append(A1_grouped[np.where(A1_grouped == w)])
        # print(channele, allele, A1_w)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.scatter(A1_raw, A1_raw, label='X ({} obs)'.format(len(A1_raw)))
        for i in range(len(A1_w)):
            plt.scatter(A1_w[i], A1_w[i] + 5, label='{}_al_{}.{} ({} obs)'.format(channel, allele, str(i), len(A1_w[i])))
        #plt.scatter(A1_l[0], A1_l[0] + 5, label='cluster_'+str(0))
        # leg = plt.legend(bbox_to_anchor=(1.3, 1))
        plt.legend(prop={'size': 6})
        plt.title(channel+"_allele_"+str(allele))
        plt.show()
sys.exit()
#######################################################################################################################
# Export results to CSV
for anno_df in all_anno_df:
    filename = output_folder_path + "/" + anno_df["channel"].iloc[0]
    anno_df.to_csv(filename, index=False)
    print("Generated: ", filename)
