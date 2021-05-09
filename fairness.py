#!/usr/bin/env python3

# Diagonising bias in predictions
import pandas as pd
import sys
from pathlib import Path
from anycache import anycache
import numpy as np
from scipy.stats import randint
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import ExtraTreeClassifier
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import LabelPowerset
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.sparse.linalg import inv

simplefilter("ignore", category=ConvergenceWarning)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


index_col = "respondent_id"
# sensitive_attributes = ["age_group", "race", "sex", "health_insurance", "marital_status", "income_poverty"]
sensitive_attributes = [
    "none",
    "marital_status",
    "income_poverty",
    "age_group",
    "race",
    "sex",
]

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def get_feature_names(meta, vartype):
    return meta[meta.type == vartype].variable


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype, meta):
        self.dtype = dtype
        self.meta = meta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        vars = get_feature_names(self.meta, self.dtype)
        return X[pd.Index(vars).intersection(X.columns).tolist()]


def get_pipeline_by_type(vartype, meta_df):
    if vartype == "ordered" or vartype == "unordered":
        return make_pipeline(
            TypeSelector(vartype, meta_df),
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(),
        )
    elif vartype == "continuous":
        return make_pipeline(
            TypeSelector(vartype, meta_df),
            SimpleImputer(strategy="median"),
            StandardScaler(),
        )
    elif vartype == "binary":
        return make_pipeline(
            TypeSelector(vartype, meta_df), SimpleImputer(strategy="most_frequent"),
        )


def preprocess(meta):
    preprocess_pipeline = make_pipeline(
        FeatureUnion(
            transformer_list=[
                ("numeric_features", get_pipeline_by_type("continuous", meta)),
                ("ordered_categorical_features", get_pipeline_by_type("ordered", meta)),
                (
                    "unordered_categorical_features",
                    get_pipeline_by_type("unordered", meta),
                ),
                ("boolean_features", get_pipeline_by_type("binary", meta)),
            ]
        )
    )
    return preprocess_pipeline


@anycache(cachedir="./cache")
def init(datadir):
    y_df = pd.read_csv(datadir / "training_set_labels.csv").set_index(index_col)
    x_df = pd.read_csv(datadir / "training_set_features.csv").set_index(index_col)
    xtest_df = pd.read_csv(datadir / "test_set_features.csv").set_index(index_col)
    meta_df = pd.read_csv(datadir / "metadata.csv")
    preprocess_pipeline = preprocess(meta_df)
    Xnew = preprocess_pipeline.fit_transform(x_df)
    Xtestnew = preprocess_pipeline.transform(xtest_df)
    # preprocess data
    return x_df, y_df, xtest_df, meta_df, preprocess_pipeline


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(
                len(set_true.union(set_pred))
            )
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def get_orth(X, S, vartype, preprocess_pipeline):
    if S is not None:
        Strain = S
        X_train_mat = preprocess_pipeline.fit_transform(X)
        Strain_mat = get_pipeline_by_type(vartype, meta).fit_transform(Strain)
        X_train_orth = (
            Strain_mat
            * inv(Strain_mat.transpose() * Strain_mat)
            * (Strain_mat.transpose() * X_train_mat)
        )
        return X_train_orth
    else:
        return preprocess_pipeline.fit_transform(X)


def get_classifier_pipelines():
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

    abc_parameters = {
        "base_estimator__max_depth": [i for i in range(2, 11, 2)],
        "base_estimator__min_samples_leaf": [5, 10],
        "n_estimators": [10, 50,],
        "learning_rate": [0.01, 0.1],
    }
    abc_tuned = RandomizedSearchCV(
        abc, abc_parameters, verbose=3, scoring="f1", n_iter=4, n_jobs=-1
    )
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {
        "max_depth": [3, None],
        "max_features": randint(1, 9),
        "min_samples_leaf": randint(1, 9),
        "criterion": ["gini", "entropy"],
    }

    # Instantiate a Decision Tree classifier: tree
    tree = DecisionTreeClassifier()

    # Instantiate the RandomizedSearchCV object: tree_cv
    tree_cv = RandomizedSearchCV(tree, param_dist, cv=5, n_jobs=8)
    return {
        "logistic": Pipeline(
            [("clf", OneVsRestClassifier(LogisticRegression(solver="sag"), n_jobs=8),),]
        ),
        "dtree": Pipeline([("clf", tree_cv)]),
        "adaboost": Pipeline([("clf", abc_tuned)]),
        "dtree_untuned": Pipeline([("clf", DecisionTreeClassifier())]),
        "svc": Pipeline([("clf", SVC(kernel="linear", C=0.1))]),
    }


def bias(X, Y, Xtest, meta, preprocess_pipeline):
    results = []
    for attr in sensitive_attributes:
        X1 = X.drop(attr, axis=1, errors="ignore")
        vartype = meta.query("variable == @attr").type.iloc[0]
        kf = KFold(n_splits=5)
        S = X[[attr]] if attr in X else None
        for index, (train_index, test_index) in enumerate(kf.split(X1)):
            X_train, X_test = X1.iloc[train_index], X1.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            x_train_orth = get_orth(
                X_train,
                S.iloc[train_index] if attr in X else None,
                vartype,
                preprocess_pipeline,
            )
            for col in Y.columns:
                y = Y_train[col]
                ytest = Y_test[col]
                # LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),])
                for modelname, ml_pipeline in get_classifier_pipelines().items():
                    ml_pipeline.fit(x_train_orth, y)
                    x_test_orth = get_orth(
                        X_test,
                        S.iloc[test_index] if attr in X else None,
                        vartype,
                        preprocess_pipeline,
                    )
                    prediction = ml_pipeline.predict(x_test_orth)
                    report = (
                        classification_report(ytest, prediction, output_dict=True),
                    )
                    print(
                        "Sensitive attribute {} Model = {} Fold = {}, Test accuracy for output {} is {}".format(
                            attr, modelname, index, col, report
                        )
                    )
                    results.append((modelname, attr, index, col, report))
    df = pd.DataFrame(
        results, columns=["model", "sensitive_attr", "cv_index", "outcome", "report"]
    )
    df = pd.concat([df, pd.json_normalize(df.report.tolist())], axis="columns").drop(
        "report", axis=1
    )
    df.drop(df.filter(".*support.*"), axis=1)
    df.to_csv("./results/bias_results.csv", index=None)
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    datadir = Path("./fludata")
    X, Y, Xtest, meta, preprocess_pipeline = init(datadir)
    bias(X, Y, Xtest, meta, preprocess_pipeline)
