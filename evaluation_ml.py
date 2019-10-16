import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
        '--to', default='evaluation_ml_res',
        help='保存评价结果的json文件名，默认是evaluation_ml_res')
    parser.add_argument('--rand_seed', default=1234, type=int)
    args = parser.parse_args()
    print(args)
    print('')

    json_res = {}
    json_res.update(deepcopy(args.__dict__))

    task_path = args.save
    kfold = StratifiedKFold(10, random_state=args.rand_seed, shuffle=True)
    kfold5 = StratifiedKFold(5, random_state=args.rand_seed, shuffle=True)
    # ----- 读取数据集 -----
    data_names = ['Ori', 'Ys', 'Rec_nobe']
    all_res = {}
    for dn in data_names:
        all_res[dn] = pd.read_csv(
            os.path.join(task_path, '%s.csv' % dn), index_col=0)
        if dn != 'Ys':
            all_res[dn] = all_res[dn].T
    for dn in ['Ori', 'Rec_nobe']:
        all_res[dn] = all_res[dn].loc[all_res['Ys'].index, :]
    for dn in data_names:
        all_res[dn] = all_res[dn].values
        # 得到的是ndarray

    # ----- 准备使用的estimator -----
    scale_rf = Pipeline([
        ('scale', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ])
    scale_pca = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(2))
    ])
    scale_pls = Pipeline([
        ('scale', StandardScaler()),
        ('pls', SelectFromModel(PLSwithVIP(3), threshold=1.)),
    ])
    scale_fdrpls_rf = Pipeline([
        ('scale', StandardScaler()),
        ('fdr', SelectFdr()),
        ('pls', SelectFromModel(PLSwithVIP(3), threshold=1.)),
        ('cls', RandomForestClassifier(100))
    ])

    # ----- subject index for select subject samples -----
    subject_index = all_res['Ys'][:, -1] == 1

    # ----- 对原始数据和去除批次后的数据关于批次的分类交叉验证 -----
    print('关于batch label的10-cv评价')
    # 使用所有的数据, original
    # 进行10-CV
    cv_res_ori = cross_val_score(
        scale_rf, all_res['Ori'], all_res['Ys'][:, 1],
        cv=kfold, scoring='accuracy', n_jobs=12)
    cv_res_nobe = cross_val_score(
        scale_rf, all_res['Rec_nobe'], all_res['Ys'][:, 1],
        cv=kfold, scoring='accuracy', n_jobs=12)
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
    ori_res_pca = scale_pca.fit_transform(all_res['Ori'])
    nobe_res_pca = scale_pca.fit_transform(all_res['Rec_nobe'])

    sil_score_ori = silhouette_score(ori_res_pca, all_res['Ys'][:, 1])
    sil_score_nobe = silhouette_score(nobe_res_pca, all_res['Ys'][:, 1])
    json_res['sil_score_all'] = {'ori': sil_score_ori, 'nobe': sil_score_nobe}
    print('所有样本轮廓系数评价')
    print('Original:')
    print(sil_score_ori)
    print('No Batch Effect:')
    print(sil_score_nobe)

    sil_score_ori = silhouette_score(
        ori_res_pca[subject_index], all_res['Ys'][subject_index, 1])
    sil_score_nobe = silhouette_score(
        nobe_res_pca[subject_index], all_res['Ys'][subject_index, 1])
    json_res['sil_score_sub'] = {'ori': sil_score_ori, 'nobe': sil_score_nobe}
    print('subject样本轮廓系数评价')
    print('Original:')
    print(sil_score_ori)
    print('No Batch Effect:')
    print(sil_score_nobe)

    sil_score_ori = silhouette_score(
        ori_res_pca[~subject_index], all_res['Ys'][~subject_index, 1])
    sil_score_nobe = silhouette_score(
        nobe_res_pca[~subject_index], all_res['Ys'][~subject_index, 1])
    json_res['sil_score_qc'] = {'ori': sil_score_ori, 'nobe': sil_score_nobe}
    print('qc样本轮廓系数评价')
    print('Original:')
    print(sil_score_ori)
    print('No Batch Effect:')
    print(sil_score_nobe)

    print('')

    # ----- 计算QC样本间的相关系数均值 -----
    print('QC数据计算平均的相关系数')
    ori_cormat = np.corrcoef(all_res['Ori'][~subject_index])
    ori_cormat_mean = ori_cormat[np.triu_indices_from(ori_cormat, k=1)].mean()
    nobe_cormat = np.corrcoef(all_res['Rec_nobe'][~subject_index])
    nobe_cormat_mean = nobe_cormat[
        np.triu_indices_from(nobe_cormat, k=1)].mean()
    json_res['qc_cor'] = {'ori_cor_mean': ori_cormat_mean,
                          'nobe_cor_mean': nobe_cormat_mean}
    print('qc样本mean of cor')
    print('Original:')
    print(ori_cormat_mean)
    print('No Batch Effect:')
    print(nobe_cormat_mean)

    print('')

    # 使用的是subject的数据
    subject_res = {k: v[subject_index, :] for k, v in all_res.items()}
    X_ori = subject_res['Ori']
    X_nobe = subject_res['Rec_nobe']
    Y = subject_res['Ys'][:, 2]

    # ----- 对原始数据和去除批次后的数据关于label的分类交叉验证 -----
    print('使用全部数据进行cancer vs no cancer的10-CV评价')
    cv_res_ori_label = cross_val_score(
        scale_rf, subject_res['Ori'], subject_res['Ys'][:, 2],
        cv=kfold, scoring='roc_auc', n_jobs=12)
    cv_res_nobe_label = cross_val_score(
        scale_rf, subject_res['Rec_nobe'], subject_res['Ys'][:, 2],
        cv=kfold, scoring='roc_auc', n_jobs=12)
    json_res['true_label_cv'] = {
        'ori': cv_res_ori_label.tolist(), 'nobe': cv_res_nobe_label.tolist()}
    print('Original:')
    print(cv_res_ori_label)
    print(np.mean(cv_res_ori_label))
    print('No Batch Effect:')
    print(cv_res_nobe_label)
    print(np.mean(cv_res_nobe_label))
    print('')

    # ----- 变量筛选再进行交叉验证 -----
    # original
    print('先做变量筛选后进行cancer vs no cancer的10-CV评价')
    cv_res_ori_label = cross_val_score(
        scale_fdrpls_rf, X_ori, Y, cv=kfold, scoring='roc_auc')
    # no batch effect
    cv_res_nobe_label = cross_val_score(
        scale_fdrpls_rf, X_nobe, Y, cv=kfold, scoring='roc_auc')

    json_res['true_label_cv_feature_selection'] = {
        'ori': cv_res_ori_label.tolist(), 'nobe': cv_res_nobe_label.tolist()}
    print('Original:')
    print(cv_res_ori_label)
    print(np.mean(cv_res_ori_label))
    print('No Batch Effect:')
    print(cv_res_nobe_label)
    print(np.mean(cv_res_nobe_label))
    print('')

    # ----- 相同数量变量的AUC评价5-CV -----
    # original
    print('使用RF-VIM排序，看不同数量的features对应的AUCs')
    scale_rf.fit(X_ori, Y)
    im_ori = scale_rf.named_steps['rf'].feature_importances_
    im_sort_indice_ori = np.argsort(im_ori)[::-1]
    scale_rf.fit(X_nobe, Y)
    im_nobe = scale_rf.named_steps['rf'].feature_importances_
    im_sort_indice_nobe = np.argsort(im_nobe)[::-1]

    features_num = np.arange(100, 1100, 100).tolist()
    ori_scores = []
    nobe_scores = []
    for fn in features_num:
        print('features number: %d' % fn)
        need_features = im_sort_indice_ori[:fn]
        X_part = X_ori[:, need_features]
        cv_res_ori_part = cross_val_score(scale_rf, X_part, Y, cv=kfold5,
                                          scoring='roc_auc', n_jobs=5)
        ori_scores.append(np.mean(cv_res_ori_part))

        need_features = im_sort_indice_nobe[:fn]
        X_part = X_nobe[:, need_features]
        cv_res_nobe_part = cross_val_score(scale_rf, X_part, Y, cv=kfold5,
                                          scoring='roc_auc', n_jobs=5)
        nobe_scores.append(np.mean(cv_res_nobe_part))

    json_res['equal_features_num'] = {
        'featurs_num': features_num,
        'ori': ori_scores,
        'nobe': nobe_scores
    }
    print('features_num')
    print(features_num)
    print('Original:')
    print(ori_scores)
    print('No Batch Effect:')
    print(nobe_scores)
    print('')


    # ----- 相同数量变量的AUC评价5-CV, 但是是完全外部的 -----
    # original
    #  print('完成外部, 使用RF-VIM排序，看不同数量的features对应的AUCs')
    #  res = {'ori': [], 'nobe': []}
    #
    #  for i in tqdm(np.arange(100, 1100, 100)):
    #      estimator = Pipeline([
    #          ('feature_select', SelectFromModel(
    #              RandomForestClassifier(100),
    #              threshold=-np.inf, max_features=i
    #          )),
    #          ('cls', RandomForestClassifier(100))
    #      ])
    #      # original
    #      cv_res = cross_val_score(estimator, X_ori, Y, cv=kfold5,
    #                               scoring="roc_auc")
    #      res['ori'].append(np.mean(cv_res))
    #      # nobe
    #      cv_res = cross_val_score(estimator, X_nobe, Y, cv=kfold5,
    #                               scoring="roc_auc")
    #      res['nobe'].append(np.mean(cv_res))
    #  json_res['equal_features_num_2'] = res
    #
    #  print('Original:')
    #  print(res['ori'])
    #  print('No Batch Effect:')
    #  print(res['nobe'])
    #  print('')


    # 保存结果的dict以json的格式保存
    with open(os.path.join(args.save, '%s.json' % args.to), 'w') as fp:
        json.dump(json_res, fp)


if __name__ == '__main__':
    main()
