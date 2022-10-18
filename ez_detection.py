import numpy as np

import scipy.io

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

   
data = scipy.io.loadmat("featureData_(14.4.2021)_NaN_handled__All-nEZ.mat")["featureData"]

def select_features(feature_type):
    assert feature_type in ("DFA", "bistability", "functional_excitation",
                            "effective_weight", "eigen_vector_centrality",
                            "cluster_coefficient", "local_efficiency")
    if feature_type=="DFA":
        return data[:, 8:12]
    if feature_type=="bistability":
        return data[:, 12:16]
    if feature_type=="functional_excitation":
        return data[:, 16:20]
    if feature_type=="effective_weight":
        return data[:, 20:26]
    if feature_type=="eigen_vector_centrality":
        return data[:, 26:32]
    if feature_type=="cluster_coefficient":
        return data[:, 32:38]
    if feature_type=="local_efficiency":
        return data[:, 38:]

def load_features(predictors):
    return np.concatenate([select_features(predictor) for predictor in predictors], axis=1)
     
predictor_names = (
    "DFA", "bistability", "functional_excitation",
    "effective_weight", "eigen_vector_centrality",
    "cluster_coefficient", "local_efficiency"
)


y = data[:, 2]
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    cv_score = []
    for features in list(powerset(predictor_names))[1:]:
        scores = []
        dataset = load_features(features)
        for seed in range(10):
            X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.2, random_state=seed)
            parameters = {'C':[0.01, .1, 1, 10, 100]}
            svc = SVC(kernel=kernel)
            clf = GridSearchCV(svc, parameters)
            clf.fit(X_train, y_train)
            d_pred = clf.decision_function(X_test)
            scores.append(roc_auc_score(y_test, d_pred))

        cv_score.append(np.mean(scores))

    np.save(kernel + "_cv_score.npy", np.array(cv_score))


