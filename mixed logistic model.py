# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 22:17:01 2021

@author: zhaoxt
"""
import sklearn
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC,LinearSVC
from scipy import interp
from sklearn.linear_model import LogisticRegression,SGDClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier,RUSBoostClassifier,EasyEnsembleClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score,r2_score,brier_score_loss,cohen_kappa_score,confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from logitboost import LogitBoost
from sklearn.manifold import TSNE
from sklearn.utils import resample
import math
sns.set(context="notebook", style="white", palette=tuple(sns.color_palette("RdBu")))
plt.rc('font',family='Times New Roman')

def _score(y_train, y_train_meta):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, y_train_meta, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    pre = (np.array(y_train_meta) >= threshold) * 1
    
    ROC = sklearn.metrics.roc_auc_score(y_train, y_train_meta)
    acc = accuracy_score(y_train, pre)
    recall = sklearn.metrics.recall_score(y_train, pre)
    precision = sklearn.metrics.precision_score(y_train, pre)
    f1 = sklearn.metrics.fbeta_score(y_train, pre, beta=1)
    average_precision = sklearn.metrics.average_precision_score(y_train, pre, pos_label=1)
    #sklearn.metrics.confusion_matrix(y_train, pre, labels=None, sample_weight=None)
    cm = confusion_matrix(y_train, pre, labels=[1, 0])
    sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
    Brier_score = brier_score_loss(y_train, pre)
    kappa = cohen_kappa_score(y_train, pre)
    r2 = r2_score(y_train, pre)
    p, r, th = sklearn.metrics.precision_recall_curve(y_train, y_train_meta, pos_label=1)
     
    return ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr
def _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity = None,std_sensitivity = None,mean_specificity = None,std_specificity = None,mean_acc = None,std_acc=None):
    f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_test_baseline.txt",'a') 
    f.write(str(ROC))
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
    f.write(str(mean_auc))
    f.write(" ")
    f.write(str(std_auc))
    f.write(" ")
    f.write(str(mean_sensitivity))
    f.write(" ")
    f.write(str(std_sensitivity))
    f.write(" ")
    f.write(str(mean_specificity))
    f.write(" ")
    f.write(str(std_specificity))
    f.write(" ")
    f.write(str(mean_acc))
    f.write(" ")
    f.write(str(std_acc))
    f.write("\n")
# =============================================================================
#     f.write(str(tprs_upper))
#     f.write("\n")
#     f.write(str(tprs_lower))
#     f.write("\n")
# =============================================================================
    f.write(str(cm))
    f.write("\n")
    f.close()

def _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_ = None,std_sensitivity_ = None,mean_specificity_ = None,std_specificity_ = None,mean_acc_ = None,std_acc_ =None):
    f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_val_baseline.txt",'a') 
    f.write(str(ROC))
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
    f.write(str(mean_auc_))
    f.write(" ")
    f.write(str(std_auc_))
    f.write(" ")
    f.write(str(mean_sensitivity_))
    f.write(" ")
    f.write(str(std_sensitivity_))
    f.write(" ")
    f.write(str(mean_specificity_))
    f.write(" ")
    f.write(str(std_specificity_))
    f.write(" ")
    f.write(str(mean_acc_))
    f.write(" ")
    f.write(str(std_acc_))
    f.write("\n")
    f.write(str(cm))
    f.write("\n")
    f.close()
# In[*]
tprs = []
aucs = []
sensitivitys = []
specificitys = []
accs = []
mean_fpr = np.linspace(0, 1, 100)
#val
tprs_ = []
aucs_ = []
sensitivitys_ = []
specificitys_ = []
accs_ = []
mean_fpr_ = np.linspace(0, 1, 100)    
y_valid_meta = np.zeros((797,1),dtype=float)
y_valid_final  = np.zeros((797,1),dtype=float)
X_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_test.csv",sep=',',header="infer")
y_test_meta = np.zeros((X_test.shape[0],1),dtype=float)
# In[*]
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv1.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv2.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv3.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv4.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv5.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv6.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv7.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv8.txt"
#name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv9.txt"
name1 = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\test_cv10.txt"
# In[*]
i=9
X_valid1 = pd.read_table(name1,sep='\t',header="infer")
cols = [c for c in X_valid1.columns if c not in ['target',"SEX"]]
valid_idx1 = X_valid1.index
y_valid1 = np.array(X_valid1)[:,2]
X_valid1 = X_valid1[cols]


X_valid = X_valid1
y_valid = y_valid1
valid_idx = valid_idx1

X_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_test.csv",sep=',',header="infer")
y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ',header = "infer")
y_test = np.array(y_test)[:,0]

coefficients = "E:\\zhaoxt_workplace\\mywork\\1慢性复杂疾病\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU\\1123_dataSummary\\mixed logistic model\\20211221mixed logistic model-10cv-coefficients.txt"
coefficients = pd.read_table(coefficients,sep='\t',header="infer")
predictions_score_valid=[]
predictions_score_test=[]

for indexs in X_valid.index:
            #X_data.loc[indexs].values[0:-1]
    predictions_score_valid.append(1/(1 + math.exp(-(np.array(coefficients)[0,i]+
                                               np.array(coefficients)[1,i]*X_valid.loc[indexs][1]+#age
                                               np.array(coefficients)[2,i]*X_valid.loc[indexs][2]+#lvh
                                               np.array(coefficients)[3,i]*X_valid.loc[indexs][3]+#hr
                                               np.array(coefficients)[4,i]*X_valid.loc[indexs][4]+#sbp
                                               np.array(coefficients)[5,i]*X_valid.loc[indexs][5]+#chd
                                               np.array(coefficients)[6,i]*X_valid.loc[indexs][6]+#vd
                                               np.array(coefficients)[7,i]*X_valid.loc[indexs][7]+
                                               np.array(coefficients)[8,i]*X_valid.loc[indexs][8]+#age
                                               np.array(coefficients)[9,i]*X_valid.loc[indexs][9]+#lvh
                                               np.array(coefficients)[10,i]*X_valid.loc[indexs][10]+#hr
                                               np.array(coefficients)[11,i]*X_valid.loc[indexs][11]+#sbp
                                               np.array(coefficients)[12,i]*X_valid.loc[indexs][12]+#chd
                                               np.array(coefficients)[13,i]*X_valid.loc[indexs][13]+#vd
                                               np.array(coefficients)[14,i]*X_valid.loc[indexs][14]+
                                               np.array(coefficients)[15,i]*X_valid.loc[indexs][15]+#age
                                               np.array(coefficients)[16,i]*X_valid.loc[indexs][16]+#lvh
                                               np.array(coefficients)[17,i]*X_valid.loc[indexs][17]+#hr
                                               np.array(coefficients)[18,i]*X_valid.loc[indexs][18]+#sbp
                                               np.array(coefficients)[19,i]*X_valid.loc[indexs][19]+#chd
                                               np.array(coefficients)[20,i]*X_valid.loc[indexs][20]+#vd
                                               np.array(coefficients)[21,i]*X_valid.loc[indexs][21]+
                                               np.array(coefficients)[22,i]*X_valid.loc[indexs][22]+#age
                                               np.array(coefficients)[23,i]*X_valid.loc[indexs][23]+#lvh
                                               np.array(coefficients)[24,i]*X_valid.loc[indexs][24]+#hr
                                               np.array(coefficients)[25,i]*X_valid.loc[indexs][25]+#sbp
                                               np.array(coefficients)[26,i]*X_valid.loc[indexs][26]+#chd
                                               np.array(coefficients)[27,i]*X_valid.loc[indexs][27]+#vd
                                               np.array(coefficients)[28,i]*X_valid.loc[indexs][28]+
                                               np.array(coefficients)[29,i]*X_valid.loc[indexs][29]+#age
                                               np.array(coefficients)[30,i]*X_valid.loc[indexs][30]#lvh
                                               ))))#diab
predictions_score_valid=np.array(predictions_score_valid)
for indexs in X_test.index:
    predictions_score_test.append(1/(1 + math.exp(-(np.array(coefficients)[0,i]+
                                               np.array(coefficients)[1,i]*X_test.loc[indexs][1]+#age
                                               np.array(coefficients)[2,i]*X_test.loc[indexs][2]+#lvh
                                               np.array(coefficients)[3,i]*X_test.loc[indexs][3]+#hr
                                               np.array(coefficients)[4,i]*X_test.loc[indexs][4]+#sbp
                                               np.array(coefficients)[5,i]*X_test.loc[indexs][5]+#chd
                                               np.array(coefficients)[6,i]*X_test.loc[indexs][6]+#vd
                                               np.array(coefficients)[7,i]*X_test.loc[indexs][7]+
                                               np.array(coefficients)[8,i]*X_test.loc[indexs][8]+#age
                                               np.array(coefficients)[9,i]*X_test.loc[indexs][9]+#lvh
                                               np.array(coefficients)[10,i]*X_test.loc[indexs][10]+#hr
                                               np.array(coefficients)[11,i]*X_test.loc[indexs][11]+#sbp
                                               np.array(coefficients)[12,i]*X_test.loc[indexs][12]+#chd
                                               np.array(coefficients)[13,i]*X_test.loc[indexs][13]+#vd
                                               np.array(coefficients)[14,i]*X_test.loc[indexs][14]+
                                               np.array(coefficients)[15,i]*X_test.loc[indexs][15]+#age
                                               np.array(coefficients)[16,i]*X_test.loc[indexs][16]+#lvh
                                               np.array(coefficients)[17,i]*X_test.loc[indexs][17]+#hr
                                               np.array(coefficients)[18,i]*X_test.loc[indexs][18]+#sbp
                                               np.array(coefficients)[19,i]*X_test.loc[indexs][19]+#chd
                                               np.array(coefficients)[20,i]*X_test.loc[indexs][20]+#vd
                                               np.array(coefficients)[21,i]*X_test.loc[indexs][21]+
                                               np.array(coefficients)[22,i]*X_test.loc[indexs][22]+#age
                                               np.array(coefficients)[23,i]*X_test.loc[indexs][23]+#lvh
                                               np.array(coefficients)[24,i]*X_test.loc[indexs][24]+#hr
                                               np.array(coefficients)[25,i]*X_test.loc[indexs][25]+#sbp
                                               np.array(coefficients)[26,i]*X_test.loc[indexs][26]+#chd
                                               np.array(coefficients)[27,i]*X_test.loc[indexs][27]+#vd
                                               np.array(coefficients)[28,i]*X_test.loc[indexs][28]+
                                               np.array(coefficients)[29,i]*X_test.loc[indexs][29]+#age
                                               np.array(coefficients)[30,i]*X_test.loc[indexs][30]#lvh
                                               ))))#diab
predictions_score_test=np.array(predictions_score_test)
    
#test
y_test_meta[:,0] += predictions_score_test    
fpr, tpr, thresholds = roc_curve(y_test, predictions_score_test)
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
threshold = thresholds[np.argmax(tpr - fpr)]
pre = (np.array(predictions_score_test) >= threshold) * 1
cm = confusion_matrix(y_test, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
acc = accuracy_score(y_test, pre)
specificitys.append(specificity) 
sensitivitys.append(sensitivity)
accs.append(acc)
#val
y_valid_meta[valid_idx-1,0] = predictions_score_valid
y_valid_final[valid_idx-1,0] = y_valid
fpr, tpr, thresholds = roc_curve(y_valid, predictions_score_valid)
tprs_.append(interp(mean_fpr_, fpr, tpr))
tprs_[-1][0] = 0.0
roc_auc_ = auc(fpr, tpr)
aucs_.append(roc_auc_)
threshold = thresholds[np.argmax(tpr - fpr)]
pre = (np.array(predictions_score_valid) >= threshold) * 1
cm = confusion_matrix(y_valid1, pre, labels=[1, 0])
sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
acc = accuracy_score(y_valid1, pre)
specificitys_.append(specificity) 
sensitivitys_.append(sensitivity)
accs_.append(acc)
# In[*]
#10-cv
y_test_meta /= float(10)
y_test_meta = y_test_meta[:,0]
y_valid_meta = y_valid_meta[:,0]   


#test
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
mean_sensitivity = np.mean(sensitivitys)
std_sensitivity = np.std(sensitivitys)
mean_specificity = np.mean(specificitys)
std_specificity = np.std(specificitys)
mean_acc = np.mean(accs)
std_acc = np.std(accs)    
#val
mean_tpr_ = np.mean(tprs_, axis=0)
mean_tpr_[-1] = 1.0
#mean_auc_ = auc(mean_fpr_, mean_tpr_)
mean_auc_ = np.mean(aucs_)
std_auc_ = np.std(aucs_)
std_tpr_ = np.std(tprs_, axis=0)
tprs_upper_ = np.minimum(mean_tpr_ + std_tpr_, 1)
tprs_lower_ = np.maximum(mean_tpr_ - std_tpr_, 0)
mean_sensitivity_ = np.mean(sensitivitys_)
std_sensitivity_ = np.std(sensitivitys_)
mean_specificity_ = np.mean(specificitys_)
std_specificity_ = np.std(specificitys_)
mean_acc_ = np.mean(accs_)
std_acc_ = np.std(accs_) 


#validation
#X = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost.csv",sep=',',header="infer")
#y_valid = np.array(X["target"])
ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_valid_final, y_valid_meta)
_validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
#test
ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
_test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)


plt.figure()
lw = 1
alpha=.8
plt.plot(fpr, tpr, color='black',lw=lw, alpha=alpha, linestyle='dashed', markersize=12,label='Mixed logistic regression:AUC = %0.3f' % ROC)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=2)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Test: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8},ncol = 1)
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\20211221deepfm_and_FHS_plot.pdf")  
