# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:22:39 2019
@author: zhaoxt
"""
# In[*]
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
sns.set(context="notebook", style="white", palette=tuple(sns.color_palette("RdBu")))
plt.rc('font',family='Times New Roman')
# In[*]
def featrue_data(datanames):
    
    data_train = "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\" + datanames + ".csv"
    data_test = "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\" + datanames + "_test.csv"
    X_train = pd.read_csv(data_train)
    cols = [c for c in X_train.columns if c not in ['ID','target']]
    X_train = X_train[cols].values
    X_test = pd.read_csv(data_test) 
    ex_list = list(X_test.ID)
    ##ex_list.remove(12049)
    X_test = X_test[X_test.ID.isin(ex_list)]
    X_test = X_test[cols].values
    
    y_train = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_UMN.txt",sep=' ')
    y_train = np.array(y_train)[:,0]
    #y_test = pd.read_table("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\pdata_test.txt",sep='\t')
    #y_test = np.array(y_test)[:,0]
    y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
    #ex_list = list(y_test.ID)
    #ex_list.remove(12049)
    y_test = y_test[y_test.ID.isin(ex_list)]
    y_test = np.array(y_test)[:,0]
    
    return X_train,X_test,y_train,y_test
# In[*]  
#Visualizing the Training Set
def plot_tsne(X_train,y_train):
    tsne = TSNE(n_components=2, random_state=0) #number components
    X_tsne = tsne.fit_transform(X_train)
    
    plt.figure(figsize=(10, 8))
    mask_0 = (y_train == 0)
    mask_1 = (y_train == 1)
    
    plt.scatter(X_tsne[mask_0, 0], X_tsne[mask_0, 1], marker='s', c='g', label='No-HF', edgecolor='k', alpha=0.7)
    plt.scatter(X_tsne[mask_1, 0], X_tsne[mask_1, 1], marker='o', c='r', label='HF', edgecolor='k', alpha=0.7)
    
    plt.title('t-SNE plot of the training data')
    plt.xlabel('1st embedding axis')
    plt.ylabel('2nd embedding axis')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()
    plt.close()
# In[*]
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
def _per():
    y =  np.arange(len(X_data))
    NUM_SPLITS = 10
    folds = []
    for i in range(NUM_SPLITS):
        random.seed( 1000 )
        train = resample(y,n_samples=len(y),replace=1,random_state=1000)
        test = np.array(list(set(y).difference(set(train))))
        folds.append(np.array([train,test]).flatten())
    
    _get = lambda x,l:[x[i] for i in l]
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
    
    y_train_meta = np.zeros((X_train.shape[0],1),dtype=float)
    y_test_meta = np.zeros((X_test.shape[0],1),dtype=float)
    return y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds

    
# In[*]
def model_svc(X_train,X_test,y_train,y_test,fun_model):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        algorithm = fun_model
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        algorithm.fit(X_resampled, y_resampled)
        y_train_meta[valid_idx,0] = algorithm.decision_function(X_valid_)
        y_test_meta_ = algorithm.decision_function(X_test)
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)
        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc


#BaggingClassifier
def model_Classifier(X_train,X_test,y_train,y_test,fun_model):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        algorithm = fun_model
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        algorithm.fit(X_resampled, y_resampled)
        y_train_meta[valid_idx,0] = algorithm.predict_proba(X_valid_)[:, 1]
        y_test_meta_ = algorithm.predict_proba(X_test)[:, 1]
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)       
        
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc

    
def XGBClassifier1_model(X_train,X_test, y_train,y_test):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        algorithm = xgb.XGBClassifier(max_depth = 8 ,min_child_weight =1,gamma = 0.4, subsample = 0.6,colsample_bytree = 0.7,eta = 0.6,silent=1,seed=1234,n_estimators = 202)
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        algorithm.fit(X_resampled, y_resampled, early_stopping_rounds=10, eval_metric="auc",eval_set=[(X_valid_, y_valid_)])
        y_train_meta[valid_idx,0] = algorithm.predict_proba(X_valid_)[:, 1]
        y_test_meta_= algorithm.predict_proba(X_test)[:, 1]
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)
        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc
    
def SGDClassifier_model(X_train,X_test, y_train,y_test):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        
        random.seed( 10 )
# =============================================================================
#         tuned_parameters = {'alpha': [10 ** a for a in range(-6, 1)],
#                                   'loss':['log','modified_huber'],
#                                   "l1_ratio":[0.15,0.3,0.5,0.7,0.9,0.95,0.99,1.0]}
#         sgd = GridSearchCV(SGDClassifier(penalty='elasticnet', 
#                                      shuffle=True, verbose=False, 
#                                      n_jobs=10, average=False),tuned_parameters, cv=10)
#         sgd.fit(X_resampled, y_resampled)
#         print(sgd.best_params_)
#         #{'alpha': 1e-06, 'l1_ratio': 0.3, 'loss': 'modified_huber'}
#         best_parameters = sgd.best_estimator_.get_params()
# =============================================================================
        random.seed( 10 )
        algorithm = SGDClassifier(loss="log", 
                      alpha= 0.001, 
                      l1_ratio=0.9,
                      penalty='elasticnet',
                      shuffle=True, 
                      verbose=False, 
                      n_jobs=10, 
                      average=False)

        
        algorithm.fit(X_resampled, y_resampled)
        y_train_meta[valid_idx,0] = algorithm.predict_proba(X_valid_)[:, 1]
        y_test_meta_ = algorithm.predict_proba(X_test)[:, 1]
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)
        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc

    
def XGBClassifier2_model(X_train,X_test, y_train,y_test):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        algorithm = xgb.XGBClassifier(objective='binary:logistic',max_depth = 6 ,min_child_weight =1,gamma = 0.4, 
                              subsample = 0.8,colsample_bytree = 0.6,eta = 0.2,silent=1,n_estimators = 50,seed=1000,nthread=4)
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        algorithm.fit(X_resampled, y_resampled, early_stopping_rounds=10, eval_metric="error",eval_set=[(X_valid_,y_valid_)])
        y_train_meta[valid_idx,0] = algorithm.predict_proba(X_valid_)[:, 1]
        y_test_meta_ = algorithm.predict_proba(X_test)[:, 1]
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)
        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc

def LogitBoost_model(X_train,X_test, y_train,y_test):
    y_train_meta,y_test_meta,_get,tprs,aucs,sensitivitys,specificitys,accs,mean_fpr,tprs_,aucs_,sensitivitys_,specificitys_,accs_,mean_fpr_,folds = _per()
    random.seed( 1000 )
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        X_train_, y_train_ = _get(X_train, train_idx), _get(y_train, train_idx)
        X_valid_, y_valid_ = _get(X_train, valid_idx), _get(y_train, valid_idx)
        algorithm = LogitBoost(n_estimators=150, random_state=0)
        X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_sample(pd.DataFrame(X_train_), pd.DataFrame(y_train_))
        algorithm.fit(X_resampled, y_resampled)
        y_train_meta[valid_idx,0] = algorithm.predict(X_valid_)
        y_test_meta_ = algorithm.predict(X_test)
        y_test_meta[:,0] += y_test_meta_
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        
        #val
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_.append(interp(mean_fpr_, fpr, tpr))
        tprs_[-1][0] = 0.0
        roc_auc_ = auc(fpr, tpr)
        aucs_.append(roc_auc_)

        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = accuracy_score(y_valid_, pre)
        specificitys_.append(specificity) 
        sensitivitys_.append(sensitivity)
        accs_.append(acc)        
        
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
    
    y_test_meta /= float(len(folds))
    y_train_meta,y_test_meta = y_train_meta[:,0],y_test_meta[:,0] 
    #validation
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_train, y_train_meta)
    _validation(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc_,std_auc_,tprs_upper_,tprs_lower_,mean_sensitivity_,std_sensitivity_,mean_specificity_,std_specificity_,mean_acc_,std_acc_)
    #test
    ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr, tpr = _score(y_test, y_test_meta)
    _test(ROC,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,mean_auc,std_auc,tprs_upper,tprs_lower,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc)
    return fpr, tpr, ROC,mean_auc,std_auc
# In[*]
    
#if __name__ == '__main__':

f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_test_baseline.txt",'a') 
f.write('====================') 
f.write("\n")
f.write('1.LinearSVC_model') 
f.write("\n")
f.write('2.BaggingClassifier_model') 
f.write("\n")
f.write('3.BalancedBaggingClassifier_model') 
f.write("\n")
f.write('4.BalancedRandomForestClassifier_model') 
f.write("\n")
f.write('5.RandomForestClassifier_model') 
f.write("\n")
f.write('6.RUSBoostClassifier_model') 
f.write("\n")
f.write('7.EasyEnsembleClassifier_model') 
f.write("\n")
f.write('8.XGBClassifier1_model') 
f.write("\n")
f.write('9.SVC_rbf_model') 
f.write("\n")
f.write('10.SVC_linear_model') 
f.write("\n")
f.write('11.SVC_poly_model') 
f.write("\n")
f.write('12.SVC_sigmoid_model') 
f.write("\n")
f.write('13.LogisticRegression_model') 
f.write("\n")
f.write('14.SGDClassifier_model') 
f.write("\n")
f.write('15.GradientBoostingClassifier_model') 
f.write("\n")
f.write('16.GraRandomForestClassifier_model') 
f.write("\n")
f.write('17.GaussianNB_model') 
f.write("\n")
f.write('18.XGBClassifier2_model') 
f.write("\n")
f.write('19.LogitBoost_model') 
f.write("\n")
f.write('auc_sensitivitys_specificitys_acc_precisions_f1_AP_Brier_score_kappa_r2_cm') 
f.write("\n")
f.close()


f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_val_baseline.txt",'a') 
f.write('====================') 
f.write("\n")
f.write('1.LinearSVC_model') 
f.write("\n")
f.write('2.BaggingClassifier_model') 
f.write("\n")
f.write('3.BalancedBaggingClassifier_model') 
f.write("\n")
f.write('4.BalancedRandomForestClassifier_model') 
f.write("\n")
f.write('5.RandomForestClassifier_model') 
f.write("\n")
f.write('6.RUSBoostClassifier_model') 
f.write("\n")
f.write('7.EasyEnsembleClassifier_model') 
f.write("\n")
f.write('8.XGBClassifier1_model') 
f.write("\n")
f.write('9.SVC_rbf_model') 
f.write("\n")
f.write('10.SVC_linear_model') 
f.write("\n")
f.write('11.SVC_poly_model') 
f.write("\n")
f.write('12.SVC_sigmoid_model') 
f.write("\n")
f.write('13.LogisticRegression_model') 
f.write("\n")
f.write('14.SGDClassifier_model') 
f.write("\n")
f.write('15.GradientBoostingClassifier_model') 
f.write("\n")
f.write('16.GraRandomForestClassifier_model') 
f.write("\n")
f.write('17.GaussianNB_model') 
f.write("\n")
f.write('18.XGBClassifier2_model') 
f.write("\n")
f.write('19.LogitBoost_model') 
f.write("\n")
f.write('auc_sensitivitys_specificitys_acc_precisions_f1_AP_Brier_score_kappa_r2_mean_auc_std_auc_cm') 
f.write("\n")
f.close()
# In[*]
#get input
#X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_dmp_xgboost")
#X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_dmp_lasso")
X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_dmp_lassoxgboost")
#X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_cor_xgboost")
#X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_cor_lasso")
#X_train,X_test,y_train,y_test = featrue_data("20201126deepfm_feature_cor_lassoxgboost")
# In[*]
#model
fpr1, tpr1, ROC1, mean_auc1, std_auc1 = model_svc(X_train,X_test,y_train,y_test,LinearSVC())
fpr2, tpr2, ROC2, mean_auc2, std_auc2 = model_Classifier(X_train,X_test, y_train,y_test,BaggingClassifier(random_state=0))
fpr3, tpr3, ROC3, mean_auc3, std_auc3 = model_Classifier(X_train,X_test, y_train,y_test,BalancedBaggingClassifier(n_estimators=50, random_state=0))
fpr4, tpr4, ROC4, mean_auc4, std_auc4 = model_Classifier(X_train,X_test, y_train,y_test,BalancedRandomForestClassifier(n_estimators=100, random_state=0))
fpr5, tpr5, ROC5, mean_auc5, std_auc5 = model_Classifier(X_train,X_test, y_train,y_test,RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1))
fpr6, tpr6, ROC6, mean_auc6, std_auc6 = model_Classifier(X_train,X_test, y_train,y_test,RUSBoostClassifier(base_estimator = AdaBoostClassifier(n_estimators=10),n_estimators=20,random_state=0))
fpr7, tpr7, ROC7, mean_auc7, std_auc7 = model_Classifier(X_train,X_test, y_train,y_test,EasyEnsembleClassifier(base_estimator = AdaBoostClassifier(n_estimators=10),n_estimators=10))
fpr8, tpr8, ROC8, mean_auc8, std_auc8 = XGBClassifier1_model(X_train,X_test, y_train,y_test)
fpr9, tpr9, ROC9, mean_auc9, std_auc9 = model_svc(X_train,X_test,y_train,y_test,SVC(kernel='rbf', C=1,probability=True,random_state=0))
fpr10, tpr10, ROC10, mean_auc10, std_auc10 = model_svc(X_train,X_test,y_train,y_test,SVC(kernel='linear', C=1,probability=True,random_state=0))
fpr11, tpr11, ROC11, mean_auc11, std_auc11 = model_svc(X_train,X_test,y_train,y_test,SVC(kernel='poly', C=1,probability=True,random_state=0))
fpr12, tpr12, ROC12, mean_auc12, std_auc12 = model_svc(X_train,X_test,y_train,y_test,SVC(kernel='sigmoid', C=1,probability=True,random_state=0))
fpr13, tpr13, ROC13, mean_auc13, std_auc13 = model_Classifier(X_train,X_test, y_train,y_test,LogisticRegression(C=1.0, penalty='l1'))
fpr14, tpr14, ROC14, mean_auc14, std_auc14 = SGDClassifier_model(X_train,X_test, y_train,y_test)
fpr15, tpr15, ROC15, mean_auc15, std_auc15 = model_Classifier(X_train,X_test, y_train,y_test,GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0))
fpr16, tpr16, ROC16, mean_auc16, std_auc16 = model_Classifier(X_train,X_test, y_train,y_test,RandomForestClassifier(n_estimators=80,random_state=0,min_samples_split=2, min_samples_leaf=2,oob_score=True))
fpr17, tpr17, ROC17, mean_auc17, std_auc17 = model_Classifier(X_train,X_test, y_train,y_test,GaussianNB())
fpr18, tpr18, ROC18, mean_auc18, std_auc18 = XGBClassifier2_model(X_train,X_test, y_train,y_test)
fpr19, tpr19, ROC19, mean_auc19, std_auc19 = LogitBoost_model(X_train,X_test, y_train,y_test)
f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_test_baseline.txt",'a') 
f.write("=========")
f.write("\n")
f.close()
f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126_val_baseline.txt",'a') 
f.write("=========")
f.write("\n")
f.close()
# In[*]
plt.figure()
lw = 1
alpha=.8

plt.plot(fpr1, tpr1, color='g',label=r'SVC-Linear:AUC = %0.2f' % ROC1, lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
plt.plot(fpr2, tpr2, color='b',label=r'Bagging:AUC = %0.2f' % ROC2 ,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
#plt.plot(fpr3, tpr3, color='black',label='c:Balanced Bagging:AUC = %0.2f' % ROC3,lw=lw, alpha=alpha)
#plt.plot(fpr4, tpr4, color='y',label=r'd:Balanced RandomForest:AUC = %0.2f' % ROC4,lw=2, alpha=alpha)
plt.plot(fpr5, tpr5, color='darkorange',label='Random Forestl:AUC = %0.2f' % ROC5,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
plt.plot(fpr6, tpr6, color='c',label=r'RUSBoost:AUC = %0.2f' % ROC6 ,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
plt.plot(fpr7, tpr7, color='m',label=r'EasyEnsemble:AUC = %0.2f' % ROC7 ,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
#plt.plot(fpr8, tpr8, color='black',label='c:XGBClassifier:AUC = %0.2f' % ROC8,lw=lw, alpha=alpha)
# =============================================================================
# plt.plot(fpr9, tpr9, color='y',label=r'd:XGBClassifier model:AUC = %0.2f' % ROC9,lw=2, alpha=alpha)
# plt.plot(fpr10, tpr10, color='darkorange',label='e:CpG model:AUC = %0.2f' % ROC10,lw=lw, alpha=alpha)
# plt.plot(fpr11, tpr11, color='r',label=r'a:SGDClassifier model:AUC = %0.2f' % ROC11 ,lw=2, alpha=alpha)
# plt.plot(fpr12, tpr12, color='b',label=r'b:Bagging model:AUC = %0.2f' % ROC12 ,lw=2, alpha=alpha)
# =============================================================================
#plt.plot(fpr13, tpr13, color='black',label='c:Logistic Regression:AUC = %0.2f' % ROC13,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
#plt.plot(fpr14, tpr14, color='y',label=r'd:SGDClassifier:AUC = %0.2f' % ROC14,lw=2, alpha=alpha)
plt.plot(fpr15, tpr15, color='grey',label='Gradient Boosting:AUC = %0.2f' % ROC15,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
#plt.plot(fpr16, tpr16, color='r',label=r'a:Random Forest:AUC = %0.2f' % ROC16 ,lw=2, alpha=alpha)
#plt.plot(fpr17, tpr17, color='k',label=r'GaussianNB:AUC = %0.2f' % ROC17 ,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
plt.plot(fpr18, tpr18, color='pink',label='XGBClassifier:AUC = %0.2f' % ROC18,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
plt.plot(fpr19, tpr19, color='y',label=r'LogitBoost:AUC = %0.2f' % ROC19,lw=lw, alpha=alpha,linestyle='dashed', markersize=12)
#come from deepFM model
plt.plot(fpr_EHRCPG, tpr_EHRCPG, color='r',label='HFrisk model:AUC = %0.2f' % ROC_EHRCPG,lw=lw, alpha=1.0,markersize=12)
#plt.plot(fpr_EHR, tpr_EHR, color='darkorange',label='e:CpG model:AUC = %0.2f' % ROC_EHR,lw=lw, alpha=alpha)
#plt.plot(fpr_CPG, tpr_CPG, color='darkorange',label='e:CpG model:AUC = %0.2f' % ROC_CPG,lw=lw, alpha=alpha)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=alpha)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Test: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8},ncol = 1)
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\1126deepfm_baseline.pdf")

# In[*]

y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
y_test = np.array(y_test)[:,0]
#y_test_meta = np.array(y_test_meta)[:,0]
#all 
ROC_EHRCPG,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr_EHRCPG, tpr_EHRCPG = _score(y_test, y_test_meta)
ROC_EHRCPG = mean_auc
#EHR
ROC_EHR,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr_EHR, tpr_EHR = _score(y_test, y_test_meta)
#ROC_EHR = mean_auc
#CPG
ROC_CPG,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr_CPG, tpr_CPG = _score(y_test, y_test_meta)
#ROC_CPG = mean_auc
# In[*]
plt.figure()
lw = 2

#come from deepFM model
#External test
plt.plot(fpr_EHRCPG, tpr_EHRCPG, color='r',label='HFrisk model:AUC = %0.2f(0.760-0.873)' % ROC_EHRCPG,lw=lw, alpha=1.0,markersize=12)
#plt.plot(fpr_EHRCPG, tpr_EHRCPG, color='r',label='HFrisk model:AUC = %0.2f(0.883-0.918)' % ROC_EHRCPG,lw=lw, alpha=1.0,markersize=12)
#plt.plot(fpr_EHR, tpr_EHR, color='g',label='5 EHR model:AUC = %0.2f(0.730-0.820)' % ROC_EHR,lw=lw, alpha=1.0,linestyle='dashed',markersize=12)
#plt.plot(fpr_CPG, tpr_CPG, color='y',label='25 CpG model:AUC = %0.2f(0.620-0.670)' % ROC_CPG,lw=lw, alpha=1.0,linestyle='dashed',markersize=12)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Simulation Test: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8})
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\0816deepfm_Simulationfeature.pdf")

# In[*]
#sex
ROC_sex1,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr_sex1, tpr_sex1 = _score(y_test, y_test_meta)
ROC_sex1 = mean_auc

ROC_sex2,acc,recall,precision,f1,sensitivity,specificity,cm,average_precision,Brier_score,kappa,r2,p,r,fpr_sex2, tpr_sex2 = _score(y_test, y_test_meta)
ROC_sex2 = mean_auc
# In[*]
#sex
plt.figure()
lw = 2

#plt.plot(fpr_sex1, tpr_sex1, color='r',label='Female: HFrisk model:AUC = %0.2f(0.915-0.944)' % ROC_sex1,lw=lw, alpha=1.0,markersize=12)
#plt.plot(fpr_sex2, tpr_sex2, color='b',label='Male: HFrisk model:AUC = %0.2f(0.861-0.879)' % ROC_sex2,lw=lw, alpha=1.0,markersize=12)
plt.plot(fpr_sex1, tpr_sex1, color='r',label='Female: HFrisk model:AUC = %0.2f(0.919-0.939)' % ROC_sex1,lw=lw, alpha=1.0,markersize=12)
plt.plot(fpr_sex2, tpr_sex2, color='b',label='Male: HFrisk model:AUC = %0.2f(0.788-0.834)' % ROC_sex2,lw=lw, alpha=1.0,markersize=12)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Test: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8})
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\0817deepfm_sex_newmodel.pdf")