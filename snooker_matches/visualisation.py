import re
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree


def calibration_plot(
    scores,
    target,
    n_bins=20
):
    binned_score = pd.qcut(scores, n_bins)
    bin_centres = sorted(pd.Series(scores).groupby(binned_score).mean().values)

    targets = target.groupby(binned_score).mean()

    score_qs = target.groupby(binned_score)\
                     .agg(['mean', 'sem'])\
                     .unstack()
    
    _, axis = plt.subplots(figsize=(10,6))
    
    # Plotting the line
    axis.plot(
        bin_centres,
        score_qs['mean'],
        color='blue'
    )
    
    # Plotting the standard error of the mean as different CI shades
    axis.fill_between(
        bin_centres, 
        score_qs['mean'] + score_qs['sem'], 
        score_qs['mean'] - score_qs['sem'],
        facecolor='blue', 
        alpha=2**1.5 / (3 ** 1.5)
    )
    axis.fill_between(
        bin_centres, 
        score_qs['mean'] + score_qs['sem'], 
        score_qs['mean'] + 2 * score_qs['sem'],
        facecolor='blue', 
        alpha=1**1.5 / (3 ** 1.5)
    )
    axis.fill_between(
        bin_centres, 
        score_qs['mean'] - score_qs['sem'], 
        score_qs['mean'] - 2 * score_qs['sem'],
        facecolor='blue', 
        alpha=1**1.5 / (3 ** 1.5)
    )

    axis.set_xlabel(scores.name)
    axis.set_ylabel(target.name)
    
    axis.plot(
        bin_centres, bin_centres, ls='--', color='black'
    )
    
    return axis



def plot_labelled_tree(estimator, features, shift=0, title=None, save=False, node_label=False):
    _, ax = plt.subplots(figsize=(18, 6))
    annotations = plot_tree(
        estimator,
        ax=ax,
        feature_names=features,
        proportion=True,
        node_ids=True,
        fontsize=10
    )

    leaf_nodes = [i for i, x in enumerate(estimator.tree_.feature) if x == -2]

    annotation_dict = {
        node_id: 'segment #{}'.format(segment+shift) for segment, node_id in enumerate(leaf_nodes)
    }

    n_nodes = estimator.tree_.node_count
    for i in range(n_nodes):
        for j in range(n_nodes-1, -1, -1):
            if node_label:
                continue
            else:
                annotations[i].set_text(
                    re.sub(
                        r'gini = 0.\d+\n', 
                        '',
                        annotations[i].get_text().replace('node #{}'.format(j), '{}'.format(
                            annotation_dict[j] if j in annotation_dict.keys() else ''
                        ))
                    )
                )
            
        if 'segment #' in annotations[i].get_text():
            annotations[i].get_bbox_patch().set_facecolor('lavender')
        else:
            annotations[i].get_bbox_patch().set_facecolor('aliceblue')
            
    if title:
        plt.title(title, fontsize=16)
        
    if save:
        plt.savefig('_'.join(title.split(' '))+'.png')
        
    plt.show()
    print('\n')