import re
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree


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