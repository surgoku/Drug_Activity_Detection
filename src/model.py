from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest,  RFE
from sklearn.feature_selection import f_regression, f_classif, mutual_info_classif, chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit


# Loading data and transforming in matrix format
# Returns: Training Data with labels, Features with Positive labels, Test data for prediction
def load_data():
    f_train = open("../train.dat", "r")
    f_test = open("../test.dat")
    X = []
    Y = []
    X_test = []

    set_lens = set()
    set_items = set()
    all_pos_class_feats = set()
    for line in f_train:
        l = line.strip().split('\t')
        x = [int(i) for i in l[1].split()]
        y = int(l[0])
        Y.append(y)
        X.append(x)
        for i in x:
            set_items.add(i)
        if y > 0:
            for i in x:
                all_pos_class_feats.add(i)
        set_lens.add(len(x))

    for line in f_test:
        l = line.strip()
        x = [int(i) for i in l.split()]
        X_test.append(x)

    items = sorted(list(set_items))
    max_feats = len(items)
    feat_index = {}

    pos_items = sorted(list(all_pos_class_feats))
    max_pos_feats = len(pos_items)
    pos_feat_index = {}

    for i,it in enumerate(items):
        feat_index[it] = i

    for i,it in enumerate(pos_items):
        if it in all_pos_class_feats:
            pos_feat_index[it] = i

    trans_X = []
    trans_pos_X = []
    print max_feats
    for x in X:
        feat = [0] * max_feats
        pos_feat = [0] * max_pos_feats
        for i in x:
            feat[feat_index[i]] = 1
            if i in pos_feat_index:
                pos_feat[pos_feat_index[i]] = 1

        trans_pos_X.append(pos_feat)
        trans_X.append(feat)

    trans_X_test = []
    trans_pos_X_test = []
    for x in X_test:
        feat = [0] * max_feats
        pos_feat = [0] * max_pos_feats
        for i in x:
            if i in feat_index:
                feat[feat_index[i]] = 1
            if i in pos_feat_index:
                pos_feat[pos_feat_index[i]] = 1

        trans_pos_X_test.append(pos_feat)
        trans_X_test.append(feat)

    return trans_X, Y, trans_X_test, trans_pos_X, trans_pos_X_test

# For selecting the edge cases (between 0 and 1 class). For handling ambiguous case. Selected based on probability obtained using XGB
def get_high_confidence_label(y_preds, y_probs):
    out_labels = []
    for l, p in zip(y_preds, y_probs):
        p = p[1]
        if p < 0.5 and l < 1:
            out_labels.append(l)
        elif p > 0.8 and l > 0:
            out_labels.append(l)
        else:
            l = np.random.randint(0, 2, 1)
            out_labels.append(l[0])
    return out_labels


# For predicting on the test set
def predict():
    # Load data
    X, Y, X_test, trans_pos_X, trans_pos_X_test = load_data()

    # PCA
    pca_model = IncrementalPCA(n_components=1000)
    pca_model.fit(X)
    X = pca_model.transform(X)
    X_test =  pca_model.transform(X_test)

    # PCA on Features with positive label
    pos_pca_model = IncrementalPCA(n_components=1000)
    pos_pca_model.fit(trans_pos_X)
    trans_pos_X = pos_pca_model.transform(trans_pos_X)
    trans_pos_X_test = pos_pca_model.transform(trans_pos_X_test)

    # Concatenatinv positive features and overall features. The idea is not to loose features imp for positive class since they are few samples.
    X = np.concatenate((X, trans_pos_X), axis=1)
    X_test = np.concatenate((X_test, trans_pos_X_test), axis=1)

    # Smote Sampling initializer
    sm = SMOTE(ratio='minority')
    X, Y = sm.fit_sample(X, Y)
    X_train = np.array(X)
    X_test = np.array(X_test)
    y_train = np.array(Y)

    # For weighted XGB
    weights = []
    for i in y_train:
        if i > 0.1:
            weights.append(1)
        else:
            weights.append(0.2)

    # Features selection
    #fs = SelectKBest(score_func=f_regression, k=500)
    #X_train = fs.fit_transform(X_train, y_train)
    #X_test = fs.transform(X_test)


    # List of models for Ensemble
    clf_lr = LogisticRegression(random_state=1)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(random_state=1)
    clf_xgb_weighs = XGBClassifier(sample_weight=weights)
    clf_svc = svm.SVC(kernel='linear')
    clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=42)
    clf_xgb = XGBClassifier()

    eclf_h = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard')

    eclf_h_w = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard', weights=[2, 2, 2, 2, 2, 5])

    eclf_s = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='soft', weights=[1, 2, 1, 1, 1, 5], flatten_transform=True)


    # Training of models
    clf = eclf_h_w
    clf.fit(X_train, y_train)
    # Predictions
    y_preds = clf.predict(X_test)



    clf_xgb.fit(X_train, y_train)
    y_probs = clf_xgb.predict_proba(X_test)
    #y_preds_xgb = clf_xgb.predict(X_test)

    # Probabilities obtained using XGB classifier to handle ambiguous cases.
    y_preds = get_high_confidence_label(y_preds, y_probs)

    f_out = open("../test_predictions.dat", "w")
    f_out_prob = open("../test_predictions_with_conf.dat", "w")
    f_out_xgb_pred = open("../test_predictions_xgb.dat", "w")

    for pred in y_preds:
        f_out.write(str(pred) + "\n")

    for prob in y_probs:
        f_out_prob.write(str(prob[0]) +"\n")


# For evaluation and experiment purposes, code description same as in predic class.
def evaluate():
    X, Y, original_X_test, trans_pos_X, trans_pos_X_test = load_data()
    del original_X_test

    trans_pos_X = np.array(trans_pos_X)
    trans_pos_X_test = np.array(trans_pos_X_test)

    X = np.array(X)
    sm = SMOTE(ratio='minority', random_state=42)

    pca_model = IncrementalPCA(n_components=1000)
    pca_model.fit(X)
    X = pca_model.transform(X)

    pos_pca_model = IncrementalPCA(n_components=1000)
    pos_pca_model.fit(trans_pos_X)
    trans_pos_X = pos_pca_model.transform(trans_pos_X)

    print X.shape, trans_pos_X.shape

    X = np.concatenate((X, trans_pos_X), axis=1)

    print X.shape

    X, Y = sm.fit_sample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #svc = SVC(kernel="linear", C=1)
    #fs = RFE(estimator=svc, n_features_to_select=100, step=3)
    #fs = SelectKBest(score_func=f_regression, k=1600)
    #X_train = fs.fit_transform(X_train, y_train)
    #X_test = fs.transform(X_test)

    weights = []
    for i in y_train:
        if i>0.1:
            weights.append(0.9)
        else:
            weights.append(0.1)

    clf_lr = LogisticRegression(random_state=1)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(random_state=1)
    clf_xgb_weighs = XGBClassifier(sample_weight=weights)
    clf_svc = svm.SVC(kernel='linear', probability=True)
    clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (50, 50), random_state = 42)
    clf_xgb = XGBClassifier()

    eclf_h = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb),  ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb) ], voting='hard')

    eclf_h_w = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard', weights=[1,2,1,1,1,5])


    eclf_s = VotingClassifier(estimators=[
         ('gnb', clf_nb),  ('svc', clf_svc), ('xgb', clf_xgb)],
        voting='soft', weights=[1,1,5])

    #eclf_h.fit(X_train, y_train)
    #eclf_s.fit(X_train, y_train)

    clf = clf_xgb
    clf.fit(X_train, y_train)

    for clf, name in zip([clf, eclf_h, eclf_h_w, eclf_s], ["XGB", "Ensemble VotingClassifier Hard", "Ensemble VotingClassifier Hard Weighted", "Ensemble VotingClassifier Soft"]):
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
        print name
        print "Accuracy: ", accuracy_score(y_test, y_preds)
        print "Average Precision, Recall, Fscore: ", precision_recall_fscore_support(y_test, y_preds, average='macro')
        print "Per class Precision, Recall, Fscore: ", precision_recall_fscore_support(y_test, y_preds)

        print
        print



if __name__ == "__main__":
    #predict()
    evaluate()


