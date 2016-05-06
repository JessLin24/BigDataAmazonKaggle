#!/usr/bin/env python
""" Amazon Access Challenge Kaggle

Jessica L & Leela K
"""

import pandas as pd
import numpy as np
from sklearn import metrics, cross_validation, linear_model, preprocessing, ensemble

SEED = 42

#Parse data (load columns 1 to 7, ignore last 2)
print "loading data"
X = pd.read_csv("data/train.csv", engine = 'python', sep = ',', header = 0, usecols=range(1, 8))
y = pd.read_csv("data/train.csv", engine = 'python', sep = ',', header = 0, usecols=[0])
y = y.as_matrix().ravel()
X_test = pd.read_csv("data/test.csv", engine = 'python', sep = ',', header = 0, usecols=range(1, 8))

# One-Hot Encoding
encoder = preprocessing.OneHotEncoder()
encoder.fit(np.vstack((X, X_test)))
X = encoder.transform(X)
X_test = encoder.transform(X_test)

# Create models
linearModel = linear_model.LogisticRegression(C=3)
forestModel = ensemble.RandomForestClassifier()
boostModel = ensemble.AdaBoostClassifier(n_estimators=100)
voteModel = ensemble.VotingClassifier(estimators=[('lm', linearModel), ('fm', forestModel), ('bm', boostModel)], voting='soft', weights=[1, 1, 1])
# Training
mean_auc = 0.0
mean_vote_auc = 0.0
n = 10
for i in range(n):
    # for each iteration, randomly hold out 30% of the data as CV set
    X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(X, y, test_size=.30, random_state=i*SEED)

    linearModel.fit(X_train, y_train) 
    linearPreds = linearModel.predict_proba(X_validate)[:, 1]
    forestModel.fit(X_train, y_train) 
    forestPreds = forestModel.predict_proba(X_validate)[:, 1]
    boostModel.fit(X_train, y_train)
    boostPreds = boostModel.predict_proba(X_validate)[:, 1]
    voteModel.fit(X_train, y_train)
    votePreds = voteModel.predict_proba(X_validate)[:, 1]
    # compute AUC metric for this CV fold
    fpr, tpr, thresholds = metrics.roc_curve(y_validate, linearPreds)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_validate, forestPreds)
    fpr3, tpr3, thresholds3 = metrics.roc_curve(y_validate, boostPreds)
    fprv, tprv, thresholdsv = metrics.roc_curve(y_validate, votePreds)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)
    vot_roc_auc = metrics.auc(fprv, tprv)
    avg_roc_auc = (roc_auc + roc_auc2 + roc_auc3)/3
    print "AUC (fold %d/%d): %f \n VoteAUC: %f \n" % (i + 1, n, avg_roc_auc, vot_roc_auc)
    mean_auc += avg_roc_auc
    mean_vote_auc += vot_roc_auc

print "Mean AUC: %f" % (mean_auc/n)
print "Mean Vote AUC: %f" % (mean_vote_auc/n)

# Prediction
linearModel.fit(X, y)
linearPreds = linearModel.predict_proba(X_test)[:, 1]
forestModel.fit(X, y)
forestPreds = forestModel.predict_proba(X_test)[:, 1]
boostModel.fit(X, y)
boostPreds = boostModel.predict_proba(X_test)[:, 1]
voteModel.fit(X, y)
votePreds = voteModel.predict_proba(X_test)[:, 1]
fPreds = []
bPreds = []
for i, bPred in enumerate(boostPreds):
    bPreds.append(bPred)
for i, fPred in enumerate(forestPreds):
    fPreds.append(fPred)
filename = raw_input("Enter name for submission file: ")
with open(filename+".csv", 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(votePreds):
                f.write("%d,%f\n" % (i + 1, pred))
