import os

import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFdr, SelectFromModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

def main():
    from copy import deepcopy
    import json

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('save', help="储存结果的文件夹")
    parser.add_argument(
        '--ica', action='store_true', help='是否使用ica处理后的数据')
    parser.add_argument(
        '--to', default='evaluation_ml_res',
        help='保存评价结果的json文件名，默认是evaluation_ml_res')
    args = parser.parse_args()
    print(args)
    print('')

    json_res = {}
    json_res.update(deepcopy(args.__dict__))

    task_path = args.save
    # ----- 读取数据集 -----
    data_names = ['original_x', 'ys', 'recons_no_batch']
    all_res = {}
    for dn in data_names:
        if dn == 'recons_no_batch' and args.ica:
            file_name = dn + '_ica'
        else:
            file_name = dn
        all_res[dn] = np.loadtxt(
            os.path.join(task_path, 'all_res_%s.txt' % file_name))

    # ----- 对原始数据和去除批次后的数据关于批次的分类交叉验证 -----
    print('关于batch label的10-cv评价')
    # 使用所有的数据, original
    # 进行10-CV
    estimator = RandomForestClassifier(n_estimators=100)
    cv_res_ori = cross_val_score(
        estimator, all_res['original_x'], all_res['ys'][:, 1],
        cv=10, scoring='accuracy', n_jobs=12)
    cv_res_nobe = cross_val_score(
        estimator, all_res['recons_no_batch'], all_res['ys'][:, 1],
        cv=10, scoring='accuracy', n_jobs=12)
    json_res['batch_label_cv'] = {
        'ori': cv_res_ori.tolist(), 'nobe': cv_res_nobe.tolist()}
    print('Original:')
    print(cv_res_ori)
    print(np.mean(cv_res_ori))
    print('No Batch Effect:')
    print(cv_res_nobe)
    print(np.mean(cv_res_nobe))
    print('')

    # ----- 使用轮廓系数来进行评价
    pca = PCA(3)
    sil_score_ori = silhouette_score(
        pca.fit_transform(all_res['original_x']), all_res['ys'][:, 1])
    sil_score_nobe = silhouette_score(
        pca.fit_transform(all_res['recons_no_batch']), all_res['ys'][:, 1])
    json_res['sil_score'] = {'ori': sil_score_ori, 'nobe': sil_score_nobe}
    print('轮廓系数评价')
    print('Original:')
    print(sil_score_ori)
    print('No Batch Effect:')
    print(sil_score_nobe)
    print('')


    # ----- 对原始数据和去除批次后的数据关于label的分类交叉验证 -----
    # 使用的是subject的数据
    subject_index = all_res['ys'][:, -1] == 1
    subject_res = {k: v[subject_index, :] for k, v in all_res.items()}
    print('使用全部数据进行cancer vs no cancer的10-CV评价')
    cv_res_ori_label = cross_val_score(
        estimator, subject_res['original_x'], subject_res['ys'][:, 2],
        cv=10, scoring='roc_auc', n_jobs=12)
    cv_res_nobe_label = cross_val_score(
        estimator, subject_res['recons_no_batch'], subject_res['ys'][:, 2],
        cv=10, scoring='roc_auc', n_jobs=12)
    json_res['true_label_cv'] = {
        'ori': cv_res_ori.tolist(), 'nobe': cv_res_nobe.tolist()}
    print('Original:')
    print(cv_res_ori_label)
    print(np.mean(cv_res_ori_label))
    print('No Batch Effect:')
    print(cv_res_nobe_label)
    print(np.mean(cv_res_nobe_label))
    print('')

    # ----- 变量筛选再进行交叉验证 -----
    estimators = Pipeline([
        ('fdr', SelectFdr()),
        ('pls', SelectFromModel(PLSwithVIP(2), threshold=1.)),
        ('cls', RandomForestClassifier(100))
    ])
    # original
    print('先做变量筛选后进行cancer vs no cancer的10-CV评价')
    X = subject_res['original_x']
    Y = subject_res['ys'][:, 2]

    cv_res_ori_label = cross_val_score(
        estimators, X, Y, cv=10, scoring='roc_auc')

    # no batch effect
    X = subject_res['recons_no_batch']
    cv_res_nobe_label = cross_val_score(
        estimators, X, Y, cv=10, scoring='roc_auc')

    json_res['true_label_cv_feature_selection'] = {
        'ori': cv_res_ori.tolist(), 'nobe': cv_res_nobe.tolist()}
    print('Original:')
    print(cv_res_ori_label)
    print(np.mean(cv_res_ori_label))
    print('No Batch Effect:')
    print(cv_res_nobe_label)
    print(np.mean(cv_res_nobe_label))
    print('')


    # 保存结果的dict以json的格式保存
    with open(os.path.join(args.save, '%s.json' % args.to), 'w') as fp:
        json.dump(json_res, fp)


if __name__ == '__main__':
    main()
