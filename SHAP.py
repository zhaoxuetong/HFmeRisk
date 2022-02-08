# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:59:40 2021

@author: zhaoxt
"""

import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as pl
# model是在第1节中训练的模型

X_train = pd.read_csv("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data/20201126deepfm_feature_dmp_lassoxgboost.csv")
cols = [c for c in X_train.columns if c not in ['ID','target']]
y_train = X_train["target"].values
X_train = X_train[cols]

#bst = xgb.XGBRegressor(eta = 0.1 , colsample_bytree = 0.5,subsample = 0.5,max_depth = 5,min_child_weigth = 3,num_boost_round = 50)
bst = xgb.XGBClassifier(objective='binary:logistic',max_depth = 6 ,min_child_weight =1,gamma = 0.4, subsample = 0.8,colsample_bytree = 0.6,eta = 0.2,silent=1,n_estimators = 50,seed=1000,nthread=4)
model = bst.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",eval_set=[(X_train, y_train)])

bst.fit(X_train, y_train)
explainer = shap.TreeExplainer(bst)

shap_values = explainer.shap_values(X_train.values)
print(shap_values.shape)

y_base = explainer.expected_value
print(y_base)
#X_train['pred'] = bst.predict(X_train[X_train.columns])
#print(X_train['pred'].mean())

j = 12
player_explainer = pd.DataFrame()
player_explainer['feature'] = X_train.columns
player_explainer['feature_value'] = X_train[X_train.columns].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer #第一列是特征名称，第二列是特征的数值，第三列是各个特征在该样本中对应的SHAP值

print('y_base + sum_of_shap_values: %.2f'%(y_base + player_explainer['shap_value'].sum()))
#print('y_pred: %.2f'%(X_train['pred'].iloc[j]))

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[j], X_train[X_train.columns].iloc[j])
shap.force_plot(explainer.expected_value, shap_values, X_train)

from matplotlib import pyplot as plt
plt.figure()
shap.summary_plot(shap_values, X_train[X_train.columns])
SUB_DIR = "D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126"
model_name = "shap"
plt.savefig("%s/plot_%s.pdf"%(SUB_DIR,model_name))
plt.close()
shap.summary_plot(shap_values, X_train[X_train.columns],plot_type="bar") #把一个特征对目标变量影响程度的绝对值的均值作为这个特征的重要性

#'Sulfonamides', 'AGE8', 'BMI8', 'CREAT8', 'Albumin_urine', 'cg00522231',
#       'cg05845376', 'cg07041999', 'cg17766026', 'cg24205914', 'cg08101977',
#       'cg25755428', 'cg05363438', 'cg13352914', 'cg03233656', 'cg05481257',
#       'cg03556243', 'cg16781992', 'cg10083824', 'cg08614290', 'cg21429551',
#       'cg00045910', 'cg10556349', 'cg21024264', 'cg27401945', 'cg06344265',
#       'cg20051875', 'cg23299445', 'cg00495303', 'cg11853697'
       
shap.dependence_plot('Sulfonamides', shap_values, X_train[X_train.columns], interaction_index=None, show=False)
shap.dependence_plot('AGE8', shap_values, X_train[X_train.columns])
shap.dependence_plot('BMI8', shap_values, X_train[X_train.columns])
shap.dependence_plot('CREAT8', shap_values, X_train[X_train.columns])
shap.dependence_plot('Albumin_urine', shap_values, X_train[X_train.columns])
shap.dependence_plot('cg00522231', shap_values, X_train[X_train.columns])
shap.dependence_plot('cg05845376', shap_values, X_train[X_train.columns])


#interaction_values
shap_interaction_values = shap.TreeExplainer(bst).shap_interaction_values(X_train[X_train.columns])
shap.summary_plot(shap_interaction_values, X_train[X_train.columns], max_display=13)

shap.dependence_plot(("Sulfonamides", "AGE8"),shap_interaction_values, X_train[X_train.columns])
shap.dependence_plot(("AGE8", "BMI8"),shap_interaction_values, X_train[X_train.columns])


tmp = np.abs(shap_interaction_values).sum(0)
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:20]
tmp2 = tmp[inds,:][:,inds]
pl.figure(figsize=(12,12))
pl.imshow(tmp2)
pl.yticks(range(tmp2.shape[0]), X_train.columns[inds], rotation=50.4, horizontalalignment="right")
pl.xticks(range(tmp2.shape[0]), X_train.columns[inds], rotation=50.4, horizontalalignment="left")
pl.gca().xaxis.tick_top()
#pl.show()
plt.savefig("%s/plot_%s.pdf"%(SUB_DIR,model_name))
plt.close()
