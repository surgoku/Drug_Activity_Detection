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
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


def balanced_sample_maker(X, y, sample_size, random_seed=None):
    """ return a balanced data set by sampling all classes with sample_size
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.iteritems():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)


def stratified_sampler(X,Y):
    sample_size = len(list(Y))
    out = balanced_sample_maker(X, Y, sample_size)
    X = out[0]
    Y = out[1]
    n = min(sum(list(Y)), len(list(Y)))
    skf = StratifiedKFold(n, shuffle=True)
    batches = []
    for _, batch in skf.split(X, Y):
        # do_something(X[batch], y[batch])
        #print X[batch], Y[batch]
        #print
        batches.append([X[batch], Y[batch]])

    return batches


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
        #for i in x:
        #    set_items.add(i)

    max_item = max(set_items)

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
    #print max_feats
    for x in X:
        #feat = [0]*max_item
        feat = [0] * max_feats
        pos_feat = [0] * max_pos_feats
        for i in x:
            #feat[i-1] = 1
            feat[feat_index[i]] = 1
            if i in pos_feat_index:
                pos_feat[pos_feat_index[i]] = 1

        trans_pos_X.append(pos_feat)
        trans_X.append(feat)

    trans_X_test = []
    trans_pos_X_test = []
    for x in X_test:
        #feat = [0]*max_item
        feat = [0] * max_feats
        pos_feat = [0] * max_pos_feats
        for i in x:
            #feat[i-1] = 1
            if i in feat_index:
                feat[feat_index[i]] = 1

            if i in pos_feat_index:
                pos_feat[pos_feat_index[i]] = 1

        trans_pos_X_test.append(pos_feat)
        trans_X_test.append(feat)


    return trans_X, Y, trans_X_test, trans_pos_X, trans_pos_X_test

def predict():
    #X, Y, X_test = load_data()
    X, Y, X_test, trans_pos_X, trans_pos_X_test = load_data()
    sm = SMOTE(ratio='minority')
    #sm = SMOTE(random_state=42, ratio='minority')
    #sm = ADASYN(random_state=42)

    #pca_model = PCA(n_components=1000)

    pca_model = IncrementalPCA(n_components=1000)
    pca_model.fit(X)
    X = pca_model.transform(X)
    X_test =  pca_model.transform(X_test)

    pos_pca_model = IncrementalPCA(n_components=1000)
    pos_pca_model.fit(trans_pos_X)
    trans_pos_X = pos_pca_model.transform(trans_pos_X)
    trans_pos_X_test = pos_pca_model.transform(trans_pos_X_test)

    X = np.concatenate((X, trans_pos_X), axis=1)
    X_test = np.concatenate((X_test, trans_pos_X_test), axis=1)

    X, Y = sm.fit_sample(X, Y)


    X_train = np.array(X)
    X_test = np.array(X_test)
    y_train = np.array(Y)

    weights = []
    for i in y_train:
        if i > 0.1:
            weights.append(1)
        else:
            weights.append(0.2)

    #fs = SelectKBest(score_func=f_regression, k=500)
    #X_train = fs.fit_transform(X_train, y_train)
    #X_test = fs.transform(X_test)


    #clf = XGBClassifier()

    clf_lr = LogisticRegression(random_state=1)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(random_state=1)
    clf_xgb_weighs = XGBClassifier(sample_weight=weights)
    clf_svc = svm.SVC(kernel='linear')
    clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50), random_state=42)

    # clf = XGBClassifier(sample_weight=weights)

    clf = XGBClassifier()
    clf_xgb = XGBClassifier()
    # clf = svm.SVC(kernel='linear', class_weight={1: 50})
    # clf = RandomForestClassifier()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (1000, 100), random_state = 42)


    # clf.fit(X_train, y_train, sample_weight=[0.5, 0.5])

    eclf_h = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard')

    eclf_h_w = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard', weights=[2, 2, 2, 2, 2, 5])

    eclf_s = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='soft', weights=[1, 2, 1, 1, 1, 5], flatten_transform=True)



    # clf.fit(X_train, y_train, sample_weight=[0.5, 0.5])
    clf = eclf_h_w
    #clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    f_out = open("../test_predictions.dat", "w")
    for i in y_preds:
        f_out.write(str(i) + "\n")


def evaluate():
    X, Y, original_X_test, trans_pos_X, trans_pos_X_test = load_data()

    trans_pos_X = np.array(trans_pos_X)
    trans_pos_X_test = np.array(trans_pos_X_test)

    X = np.array(X)

    sm = SMOTE(ratio='minority', random_state=42)
    #sm = ADASYN(random_state=42)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=25)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    batches = stratified_sampler(X_train, y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    xgb_batch_models =[]
    for batch in batches:
        m = xgb.DMatrix(batch[0], label=batch[1])
        xgb_batch_models.append(m)
    params = {'objective': 'reg:linear', 'verbose': False}

    last_mod = None
    for mod in xgb_batch_models:
        m = xgb.train(params, mod, 30, xgb_model=last_mod)
        params.update({'process_type': 'update',
                       'updater': 'refresh',
                       'refresh_leaf': True})
        last_mod = m


    y_preds = last_mod.predict(xg_test)

    precision_recall_fscore_support(y_test, y_preds)

    exit()

    del original_X_test

    #pca_model = PCA(n_components=1000)

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=25)



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

    #clf = XGBClassifier(sample_weight=weights)

    clf = XGBClassifier()
    clf_xgb = XGBClassifier()
    #clf = svm.SVC(kernel='linear', class_weight={1: 50})
    #clf = RandomForestClassifier()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (1000, 100), random_state = 42)


    # clf.fit(X_train, y_train, sample_weight=[0.5, 0.5])

    eclf_h = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb),  ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb) ], voting='hard')

    eclf_h_w = VotingClassifier(estimators=[
        ('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_nb), ('svc', clf_svc), ('mlp', clf_mlp), ('xgb', clf_xgb)],
        voting='hard', weights=[1,2,1,1,1,5])


    eclf_s = VotingClassifier(estimators=[
         ('gnb', clf_nb),  ('svc', clf_svc), ('xgb', clf_xgb)],
        voting='soft', weights=[2,2,5])

    #eclf_h.fit(X_train, y_train)
    #eclf_s.fit(X_train, y_train)

    for clf in [clf, eclf_h, eclf_h_w, eclf_s]:
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)

        print accuracy_score(y_test, y_preds)
        print precision_recall_fscore_support(y_test, y_preds, average='macro')
        print precision_recall_fscore_support(y_test, y_preds)




if __name__ == "__main__":
    evaluate()
    #predict()

