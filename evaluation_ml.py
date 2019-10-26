import os
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFdr, SelectFromModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.decomposition import PCA


def vip(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j]))**2
            for j in range(h)
        ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips


class PLSwithVIP(PLSRegression):
    def __init__(
        self, n_components=2, scale=True, max_iter=500, tol=1e-6, copy=True
    ):
        super(PLSwithVIP, self).__init__(
            n_components, scale, max_iter, tol, copy)

    def fit(self, X, Y):
        res = super(PLSwithVIP, self).fit(X, Y)
        self.feature_importances_ = vip(self)
        return res


def mean_cor(arr):
    cormat = np.corrcoef(arr)
    return cormat[np.triu_indices_from(cormat, k=1)].mean()


def mean_distance(arr):
    distlist = dist.pdist(arr)
    return np.mean(distlist)


def one_fold(selects, cls, X, Y, train, test, score):
    new_selects = deepcopy(selects)
    new_cls = deepcopy(cls)
    Y_train, Y_test = Y[train], Y[test]
    X_train, X_test = X[train], X[test]
    select_mask = []
    for select in new_selects:
        select.fit(X_train, Y_train)
        select_mask.append(select.get_support())
    select_mask = np.stack(select_mask, axis=1).all(axis=1)
    new_cls.fit(X_train[:, select_mask], Y_train)
    pred_test = new_cls.predict_proba(X_test[:, select_mask])
    score_test = score(Y_test, pred_test[:, -1])
    return score_test, int(select_mask.sum())


def SelectClsCv(selects, cls, kfold, X, Y, score=roc_auc_score, n_jobs=10):
    pool = Pool(processes=n_jobs)
    results = []
    for train, test in kfold.split(X, Y):
        results.append(
            pool.apply_async(
                one_fold, (selects, cls, X, Y, train, test, score)
            )
        )
    pool.close()
    pool.join()

    scores = []
    feature_nums = []
    for res in results:
        ss, ff = res.get()
        scores.append(ss)
        feature_nums.append(ff)
    return scores, feature_nums


def main():
    from copy import deepcopy
    import json

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', help="储存结果的文件夹")
    parser.add_argument(
        '--to', default='evaluation_ml_res',
        help='保存评价结果的json文件名，默认是evaluation_ml_res')
    parser.add_argument('--rand_seed', default=1234, type=int)
    parser.add_argument('--estimator', default='rf')
    parser.add_argument("--ori_eval", action="store_true")
    parser.add_argument("--njobs", default=10)
    args = parser.parse_args()
    print(args)
    print('')

    # ----- 准备保存数据的dict -----
    json_res = {}
    json_res.update(deepcopy(args.__dict__))

    # ----- 读取数据集 -----
    scale = StandardScaler()
    task_path = args.save
    data_names = ['Ori', "Rec_nobe"]
    Y_name = "Ys"
    Y = pd.read_csv(os.path.join(task_path, "%s.csv" % Y_name), index_col=0)
    dats = {}
    unscale_dats = {}
    for dn in data_names:
        dat = pd.read_csv(
            os.path.join(task_path, '%s.csv' % dn), index_col=0).T
        dat = dat.loc[Y.index, :]
        dat = dat.values  # 得到的是ndarray
        unscale_dats[dn] = dat  # 保存没有进行scale的数据，做sil score和cor
        dats[dn] = scale.fit_transform(dat)  # 后面都要用到，这里直接标准化了

    # subject index for select subject samples
    subject_index = (Y.loc[:, "class"] == 1).values

    # 后面会只使用subject的数据，这里先准备一下
    sub_dats = {k: v[subject_index, :] for k, v in dats.items()}
    sub_Y = Y.loc[subject_index, "group"].values

    # ----- 准备使用的estimator -----
    kfold = StratifiedKFold(10, random_state=args.rand_seed, shuffle=True)
    kfold5 = StratifiedKFold(5, random_state=args.rand_seed, shuffle=True)
    cls_estimator = RandomForestClassifier(500, random_state=args.rand_seed) \
        if args.estimator == 'rf' else SVC(probability=True)
    cls = cls_estimator
    pca = PCA(3)
    fdr = SelectFdr()
    pls = SelectFromModel(PLSwithVIP(3), threshold=1.)
    pls_noselect = PLSRegression(3)

    # ----- 对原始数据和去除批次后的数据关于批次的分类交叉验证 -----
    print('关于batch label的10-cv评价')
    # 使用所有的数据, original
    # 进行10-CV
    json_res["batch_label_cv"] = {}
    for dn, dat in dats.items():
        cv_res = cross_val_score(
            cls, dat, Y.loc[:, "batch"].values,
            cv=kfold, scoring='accuracy', n_jobs=args.njobs)
        json_res['batch_label_cv'][dn] = cv_res.tolist()
        print('%s:' % dn)
        print(cv_res)
        print(np.mean(cv_res))
    print('')

    # ----- 使用轮廓系数来进行评价
    print('计算关于batch label的轮廓系数')
    json_res['sil_score_all'] = {}
    json_res['sil_score_sub'] = {}
    json_res['sil_score_qc'] = {}
    for dn, dat in unscale_dats.items():
        pca_res = pca.fit_transform(dat)
        sil_score_all = silhouette_score(pca_res, Y.loc[:, "batch"].values)
        sil_score_sub = silhouette_score(
            pca_res[subject_index], Y.loc[subject_index, "batch"].values)
        sil_score_qc = silhouette_score(
            pca_res[~subject_index], Y.loc[~subject_index, "batch"].values)
        print("%s: " % dn)
        print("For all sample: %.4f" % sil_score_all)
        print("For subject sample: %.4f" % sil_score_sub)
        print("For qc sample: %.4f" % sil_score_qc)

        json_res['sil_score_all'][dn] = sil_score_all
        json_res['sil_score_sub'][dn] = sil_score_sub
        json_res['sil_score_qc'][dn] = sil_score_qc
    print('')

    # ----- 计算样本间的距离均值 -----
    print('计算在3维主成分空间上的样本平均距离')
    json_res['dist_all'] = {}
    json_res['dist_sub'] = {}
    json_res['dist_qc'] = {}
    for dn, dat in dats.items():
        pca_dat = pca.fit_transform(dat)
        all_dist = mean_distance(pca_dat)
        sub_dist = mean_distance(pca_dat[subject_index])
        qc_dist = mean_distance(pca_dat[~subject_index])
        print("%s: " % dn)
        print("For all sample: %.4f" % all_dist)
        print("For subject sample: %.4f" % sub_dist)
        print("For qc sample: %.4f" % qc_dist)

        json_res['dist_all'][dn] = all_dist
        json_res['dist_sub'][dn] = sub_dist
        json_res['dist_qc'][dn] = qc_dist
    print('')

    # ----- 计算样本间的相关系数均值 -----
    print('计算样本间的平均相关系数')
    json_res['cor_all'] = {}
    json_res['cor_sub'] = {}
    json_res['cor_qc'] = {}
    for dn, dat in unscale_dats.items():
        all_cor = mean_cor(dat)
        sub_cor = mean_cor(dat[subject_index])
        qc_cor = mean_cor(dat[~subject_index])
        print("%s: " % dn)
        print("For all sample: %.4f" % all_cor)
        print("For subject sample: %.4f" % sub_cor)
        print("For qc sample: %.4f" % qc_cor)

        json_res['cor_all'][dn] = all_cor
        json_res['cor_sub'][dn] = sub_cor
        json_res['cor_qc'][dn] = qc_cor
    print('')

    # ----- 对原始数据和去除批次后的数据关于label的分类交叉验证 -----
    print('使用全部数据进行cancer vs no cancer的10-CV评价')
    json_res['group_label_cv'] = {}
    for dn, dat in sub_dats.items():
        cv_res = cross_val_score(
            cls, dat, sub_Y, cv=kfold,
            scoring='roc_auc', n_jobs=args.njobs)
        json_res['group_label_cv'][dn] = cv_res.tolist()
        print("%s: " % dn)
        print(cv_res)
        print(np.mean(cv_res))
    print('')

    # ----- 变量筛选再进行交叉验证 -----
    print('先做变量筛选后进行cancer vs no cancer的10-CV评价')
    json_res['group_selected_cv'] = {}
    json_res["group_selected_features"] = {}
    for dn, dat in sub_dats.items():
        cv_res, num_features = SelectClsCv(
            [fdr, pls], cls, kfold, dat, sub_Y, score=roc_auc_score,
            n_jobs=args.njobs)
        json_res['group_selected_cv'][dn] = cv_res
        json_res['group_selected_features'][dn] = num_features
        print("%s: " % dn)
        print(cv_res)
        print(np.mean(cv_res))
        print(num_features)
        print(np.mean(num_features))
    print("")

    # ----- 变量筛选再进行交叉验证Deng -----
    print('先做变量筛选后(deng)进行cancer vs no cancer的10-CV评价')
    json_res["group_selected_cv_deng"] = {}
    json_res["group_selected_features_deng"] = {}
    for dn, dat in sub_dats.items():
        fdr.fit(dat, sub_Y)
        pls.fit(dat, sub_Y)
        select_mask = fdr.get_support() & pls.get_support()
        cv_res = cross_val_score(
            cls, dat[:, select_mask], sub_Y,
            cv=kfold, scoring='roc_auc', n_jobs=args.njobs)
        json_res['group_selected_cv_deng'][dn] = cv_res.tolist()
        json_res["group_selected_features_deng"][dn] = int(select_mask.sum())
        print("%s: " % dn)
        print(cv_res)
        print(np.mean(cv_res))
        print(int(select_mask.sum()))
    print("")

    # ----- 相同数量变量的AUC评价5-CV -----
    print('使用PLS-VIP排序，看不同数量的features对应的AUCs')
    f_nums = np.arange(100, 1100, 100).tolist()
    json_res["group_fixed_cv"] = {}
    for dn, dat in sub_dats.items():
        pls_noselect.fit(dat, sub_Y)
        vip_value = np.argsort(vip(pls_noselect))[::-1]
        cv_reses = []
        for fn in f_nums:
            cv_res = cross_val_score(
                cls, dat[:, vip_value[:fn]], sub_Y, cv=kfold5,
                scoring='roc_auc', n_jobs=args.njobs)
            cv_reses.append(cv_res.tolist())
        json_res['group_fixed_cv'][dn] = cv_reses
        print("%s: " % dn)
        print([np.mean(cv_res) for cv_res in cv_reses])

    # 保存结果的dict以json的格式保存
    with open(os.path.join(task_path, '%s.json' % args.to), 'w') as fp:
        json.dump(json_res, fp)


if __name__ == '__main__':
    main()
