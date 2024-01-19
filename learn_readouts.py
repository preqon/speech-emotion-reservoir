'''
This script trains a SVM classifier to map Empath's readouts with the learned 
weights to speech emotion.
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

def main():

    #testing first four features only
    X = np.zeros((3000,20))
    y = np.zeros(3000)

    #testing first four features only
    X_test = np.zeros((871, 20)) 
    y_test = np.zeros(871)

    sample_idx = 0
    for sample_fname in glob.glob('final_readouts/2024-01-18/*pk'):
        with open(sample_fname, 'rb') as f:
            sample_data = pickle.load(f) 

        #testing first four features only
        sample_data = sample_data[:,30:34]

        sample_data = sample_data.flatten() 
        sample_label = sample_fname.split('/')[-1].split('_')[2] 
        sample_label = labels[sample_label]
        if sample_idx < 3000:
            X[sample_idx] = sample_data
            y[sample_idx] = sample_label
        else:
            X_test[3000-sample_idx] = sample_data
            y_test[3000-sample_idx] = sample_label
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