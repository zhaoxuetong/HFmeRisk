# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:27:32 2019

@author: zhaoxt
"""
# In[*]
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy import interp
from collections import Counter
from scipy import stats
import xgboost as xgb
from xgboost import plot_importance
from sklearn import svm
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import seaborn as sns
sns.set(context="notebook", style="white", palette=tuple(sns.color_palette("RdBu")))
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import ElasticNetCV,LogisticRegression,SGDClassifier ,SGDRegressor
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingClassifier,IsolationForest
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_curve, auc,accuracy_score
from imblearn.metrics import sensitivity_specificity_support,sensitivity_score,specificity_score,geometric_mean_score
from imblearn.metrics import geometric_mean_score as gmean
from sklearn.metrics import confusion_matrix, mean_squared_error
from imblearn.metrics import make_index_balanced_accuracy as iba
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade,EasyEnsemble, BalancedBaggingClassifier,BalancedRandomForestClassifier,RUSBoostClassifier,EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
#from GradientBoostingClassifier import plot_importance
#from sklearn.decomposition import NMF
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from logitboost import LogitBoost
from sklearn.manifold import TSNE
from sklearn.metrics import brier_score_loss

import pickle
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import roc_curve, auc, accuracy_score
from imblearn.metrics import sensitivity_specificity_support,sensitivity_score,specificity_score,geometric_mean_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
plt.rc('font',family='Times New Roman')
# In[*]

def score(roc_auc,acc,sensitivity,specificity,recall,precision,f1,cm,average_precision,Brier_score,kappa,r2):
    f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_test_baseline.txt",'a') 
    f.write(str(roc_auc))
    f.write(" ")
    f.write(str(sensitivity))
    f.write(" ")
    f.write(str(specificity))
    f.write(" ")
    f.write(str(acc))
    f.write(" ")
    f.write(str(precision))
    f.write(" ")
    f.write(str(f1))
    f.write(" ")
    f.write(str(average_precision))
    f.write(" ")
    f.write(str(Brier_score))
    f.write(" ")
    f.write(str(kappa))
    f.write(" ")
    f.write(str(r2))
    f.write(" ")
    f.write(str(cm))
    f.write("\n")
    f.write("===================")
    f.write("\n")
    f.close()
# In[*] 
#mixed logistic model
X_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_test.csv",sep=',',header="infer")
y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ',header = "infer")
y_test = np.array(y_test)[:,0]
    
predictions_score=[]
for indexs in X_test.index:
    #X_data.loc[indexs].values[0:-1]
    predictions_score.append(1/(1 + math.exp(-(-15.6338797+
                                               0.7634597*X_test.loc[indexs][1]+#age
                                               7.3748606*X_test.loc[indexs][2]+#lvh
                                               5.8339483*X_test.loc[indexs][3]+#hr
                                               6.5574935*X_test.loc[indexs][4]+#sbp
                                               4.0489741*X_test.loc[indexs][5]+#chd
                                               11.4542900*X_test.loc[indexs][6]+#vd
                                               4.9044595*X_test.loc[indexs][7]-
                                               5.9617000*X_test.loc[indexs][8]+#age
                                               6.3350521*X_test.loc[indexs][9]+#lvh
                                               10.8487157*X_test.loc[indexs][10]+#hr
                                               9.2456837*X_test.loc[indexs][11]+#sbp
                                               1.1094021*X_test.loc[indexs][12]+#chd
                                               10.4195156*X_test.loc[indexs][13]+#vd
                                               2.5393813*X_test.loc[indexs][14]-
                                               2.9114656*X_test.loc[indexs][15]-#age
                                               2.5936079*X_test.loc[indexs][16]-#lvh
                                               4.0525313*X_test.loc[indexs][17]+#hr
                                               13.6399520*X_test.loc[indexs][18]-#sbp
                                               7.8934985*X_test.loc[indexs][19]-#chd
                                               5.0566821*X_test.loc[indexs][20]-#vd
                                               2.2829597*X_test.loc[indexs][21]-
                                               5.4800662*X_test.loc[indexs][22]-#age
                                               3.6894219*X_test.loc[indexs][23]+#lvh
                                               9.1943155*X_test.loc[indexs][24]+#hr
                                               7.0205468*X_test.loc[indexs][25]-#sbp
                                               3.6061284*X_test.loc[indexs][26]-#chd
                                               12.3884444*X_test.loc[indexs][27]-#vd
                                               2.4002814*X_test.loc[indexs][28]-
                                               4.6786293*X_test.loc[indexs][29]+#age
                                               6.7084474*X_test.loc[indexs][30]#lvh                                               
                                               ))))#diab
predictions_score=np.array(predictions_score)

fpr1, tpr1, thresholds = sklearn.metrics.roc_curve(y_test, predictions_score, pos_label=1)
threshold = thresholds[np.argmax(tpr1 - fpr1)]
pre = (np.array(predictions_score) >= threshold) * 1
roc_auc1 = sklearn.metrics.roc_auc_score(y_test, predictions_score)
acc = sklearn.metrics.accuracy_score(y_test, pre)
sklearn.metrics.confusion_matrix(y_test, pre, labels=None, sample_weight=None)
cm = confusion_matrix(y_test, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
precision = sklearn.metrics.precision_score(y_test, pre)
f1 = sklearn.metrics.fbeta_score(y_test, pre, beta=1)
average_precision = sklearn.metrics.average_precision_score(y_test, pre, pos_label=1)
Brier_score = brier_score_loss(y_test, pre)
recall = sklearn.metrics.recall_score(y_test, pre)
kappa = cohen_kappa_score(y_test, pre)
r2 = r2_score(y_test, pre)
print(sensitivity, specificity,acc)

score(roc_auc1,acc,sensitivity,specificity,recall,precision,f1,cm,average_precision,Brier_score,kappa,r2)  

plt.figure()
lw = 1
alpha=.8


plt.plot(fpr1, tpr1, color='black',lw=lw, alpha=alpha, linestyle='dashed', markersize=12,label='Mixed logistic regression:AUC = %0.3f' % roc_auc1)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=2)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Test: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8},ncol = 1)
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\20211221deepfm_and_FHS_plot.pdf")  
# In[*] 
#man

X_data = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_FHSman_test.csv",sep=' ',header="infer")
y_data = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_man.txt",sep=' ')
y_data = np.array(y_data)[:,0]

predictions_score=[]
for indexs in X_data.index:
    #X_data.loc[indexs].values[0:-1]
    predictions_score.append(1/(1 + math.exp(-(-9.2087+
                                               0.0412*X_data.loc[indexs][5]+#age
                                               0.9026*X_data.loc[indexs][2]+#lvh
                                               0.0166*X_data.loc[indexs][0]+#hr
                                               0.00804*X_data.loc[indexs][6]+#sbp
                                               1.6079*X_data.loc[indexs][3]+#chd
                                               0.9714*X_data.loc[indexs][1]+#vd
                                               0.2244*X_data.loc[indexs][4]))))#diab
predictions_score=np.array(predictions_score)

fpr1, tpr1, thresholds = sklearn.metrics.roc_curve(y_data, predictions_score, pos_label=1)
threshold = thresholds[np.argmax(tpr1 - fpr1)]
pre = (np.array(predictions_score) >= threshold) * 1
roc_auc1 = sklearn.metrics.roc_auc_score(y_data, predictions_score)
acc = sklearn.metrics.accuracy_score(y_data, pre)
sklearn.metrics.confusion_matrix(y_data, pre, labels=None, sample_weight=None)
cm = confusion_matrix(y_data, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
Brier_score = brier_score_loss(y_data, pre)
print(sensitivity, specificity,acc)

score(roc_auc1,acc,sensitivity,specificity)
# In[*]
#woman
X_data = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_FHSwoman_test.csv",sep=' ',header="infer")
y_data = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_woman.txt",sep=' ')
y_data = np.array(y_data)[:,0]


#woman
predictions_score=[]
for indexs in X_data.index:
    #X_data.loc[indexs].values[0:-1]
    predictions_score.append(1/(1 + math.exp(-(-10.7988+
                                               0.0503*X_data.loc[indexs][5]+
                                               1.3402*X_data.loc[indexs][2]+
                                               0.0105*X_data.loc[indexs][0]+
                                               0.00337*X_data.loc[indexs][7]+
                                               1.5549*X_data.loc[indexs][3]+
                                               1.3929*X_data.loc[indexs][1]+
                                               1.3857*X_data.loc[indexs][4]+
                                               0.0578*X_data.loc[indexs][6]-
                                               0.9860*X_data.loc[indexs][8]))))
predictions_score=np.array(predictions_score)


fpr2, tpr2, thresholds = sklearn.metrics.roc_curve(y_data, predictions_score, pos_label=1)
threshold = thresholds[np.argmax(tpr2 - fpr2)]
pre = (np.array(predictions_score) >= threshold) * 1
roc_auc2 = sklearn.metrics.roc_auc_score(y_data, predictions_score)
acc = sklearn.metrics.accuracy_score(y_data, pre)
cm = confusion_matrix(y_data, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
Brier_score = brier_score_loss(y_data, pre)

score(roc_auc2,acc,sensitivity,specificity)
# In[*]

#man
y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_man.txt",sep=' ')
y_test = np.array(y_test)[:,0]

fpr3, tpr3, thresholds = sklearn.metrics.roc_curve(y_test, y_test_meta, pos_label=1)
threshold = thresholds[np.argmax(tpr3 - fpr3)]
pre = (np.array(y_test_meta) >= threshold) * 1
roc_auc3 = sklearn.metrics.roc_auc_score(y_test, y_test_meta)
acc = sklearn.metrics.accuracy_score(y_test, pre)
sklearn.metrics.confusion_matrix(y_test, pre, labels=None, sample_weight=None)
cm = confusion_matrix(y_test, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
Brier_score = brier_score_loss(y_test, pre)

score(roc_auc3,acc,sensitivity,specificity)
# In[*]


#woman
y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_woman.txt",sep=' ')
y_test = np.array(y_test)[:,0]

fpr4, tpr4, thresholds = sklearn.metrics.roc_curve(y_test, y_test_meta, pos_label=1)
threshold = thresholds[np.argmax(tpr4 - fpr4)]
pre = (np.array(y_test_meta) >= threshold) * 1
roc_auc4 = sklearn.metrics.roc_auc_score(y_test, y_test_meta)
acc = sklearn.metrics.accuracy_score(y_test, pre)
sklearn.metrics.confusion_matrix(y_test, pre, labels=None, sample_weight=None)
cm = confusion_matrix(y_test, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
Brier_score = brier_score_loss(y_test, pre)

score(roc_auc4,acc,sensitivity,specificity)
# In[*]  
##############
    
plt.figure()
lw = 2


plt.plot(fpr1, tpr1, color='y',lw=lw, label='a:FHS male(n=52):AUC = %0.3f' % roc_auc1)
plt.plot(fpr2, tpr2, color='b',lw=lw, label='b:FHS female model(n=27):AUC = %0.3f' % roc_auc2)
plt.plot(fpr3,tpr3, color='r',label=r'c:HFrisk male model(n=52):AUC = %0.3f' % roc_auc3 ,lw=2, alpha=.8)
plt.plot(fpr4,tpr4, color='g',label=r'd:HFrisk female model(n=27):AUC = %0.3f' % roc_auc4 ,lw=2, alpha=.8)

#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)
plt.xlim([-0.1, 1.1])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right",fontsize = "medium")
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\1126deepfm_and_FHS_plot.pdf")