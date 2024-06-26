'''
This script trains a SVM classifier to map Empath's readouts with the learned 
weights to speech emotion (or digits in the case of fsdd).
'''

#set up samples: extract labels + train set and test set.
#train SVM

# - SAD - sadness - 0 
# - ANG - angry - 1
# - DIS - disgust - 2
# - FEA - fear - 3
# - HAP - happy - 4
# - NEU - neutral - 5

# X has shape (n_samples, n_features)
# y has shape (n_samples)


import glob
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC 
import sys

# =====
labels = { 
    'SAD': 0,
    'ANG': 1,
    'DIS': 2,
    'FEA': 3,
    'HAP': 4,
    'NEU': 5,
    }
# ====
dstring = "2024-04-19"

def main():
    dataset = sys.argv[1]
    if dataset== "crema":
        n_train_samples = 6000 
        n_total_samples = 7000
        n_pools = 250
    elif dataset == "fsdd":
        n_train_samples = 2500
        n_total_samples = 3000
        n_pools = 200

    X = np.zeros((n_train_samples,n_pools))
    y = np.zeros(n_train_samples)

    X_test = np.zeros((n_total_samples-n_train_samples, n_pools)) 
    y_test = np.zeros(n_total_samples-n_train_samples)

    sample_idx = 0
    for sample_fname in glob.glob(f'final_readouts/{dstring}/*pk'):
        with open(sample_fname, 'rb') as f:
            sample_data = pickle.load(f) 

        sample_data = sample_data.flatten() 
        if dataset == "crema":
            sample_label = sample_fname.split('/')[-1].split('_')[2] 
            sample_label = labels[sample_label]
        elif dataset == "fsdd":
            sample_label = int(sample_fname.split('/')[-1].split('_')[0])
        if sample_idx < n_train_samples:
            X[sample_idx] = sample_data
            y[sample_idx] = sample_label
        else:
            X_test[n_train_samples-sample_idx] = sample_data
            y_test[n_train_samples-sample_idx] = sample_label
        sample_idx += 1

    clf =  make_pipeline(
        StandardScaler(),
        LinearSVC(
            dual='auto',
            C=1.0
            )
        )
    clf.fit(X, y)
    score = clf.score(X_test, y_test)
    print(f"Mean accuracy: {score*100:.2f}%")
if __name__ == "__main__":
    main()
