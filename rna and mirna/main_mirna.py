import os
import numpy as np
import pandas as pd
import random
import sklearn
import tensorflow as tf
from sklearn.metrics import make_scorer,roc_auc_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
import config_mirna as config
from metrics import gini_norm
from DeepFM import DeepFM
from numpy.random import seed
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
#seed(2020)
from tensorflow import set_random_seed
#set_random_seed(2020) 
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
import shap
plt.rc('font',family='Times New Roman')        


## In[*]
def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    ex_list = list(dfTest.ID)
    dfTest = dfTest[dfTest.ID.isin(ex_list)]

    cols = [c for c in dfTrain.columns if c not in ['ID','target']]
    #cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['ID'].values

    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices,ex_list,cols

def run_base_model_dfm(dfTrain,dfTest,folds,dfm_params,ex_list):
    fd = FeatureDictionary(dfTrain=dfTrain,dfTest=dfTest, numeric_cols=config.NUMERIC_COLS,ignore_cols = config.IGNORE_COLS)
    data_parser = DataParser(feat_dict= fd)
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_test,Xv_test,ids_test = data_parser.parse(df=dfTest)

    print(dfTrain.dtypes)

    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0],1),dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0],1),dtype=float)

    _get = lambda x,l:[x[i] for i in l]

    gini_results_cv = np.zeros(len(folds),dtype=float)
    gini_results_cv_test = np.zeros(len(folds),dtype=float)
    gini_results_epoch_train = np.zeros((len(folds),dfm_params['epoch']),dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds),dfm_params['epoch']),dtype=float)
    gini_results_epoch_test = np.zeros((len(folds),dfm_params['epoch']),dtype=float)
    #y_test = pd.read_table("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\pdata_test.txt",sep='\t')
    #y_test = pd.read_table("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\FHS\\pdata_woman_FHS_model.txt",sep='\t')
    y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\mirna\\deepfm_pdata_JHU.txt",sep=' ')
#    y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
    y_test = y_test[y_test.ID.isin(ex_list)]
    #y_test = y_test['JHU_DMP_new.chf']
    #y_test = np.array(y_test)[:,0]
    #y_test = np.array(y_test['HFpEF.chf'].values.tolist())
    y_test = np.array(y_test['test_mirna.Chf'].values.tolist())
    
    random.seed( 1000 )
    tprs = []
    aucs = []
    tprs_val = []
    aucs_val = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr_val = np.linspace(0, 1, 100)
    loss_batch = []
    sensitivitys = []
    specificitys = []
    accs = []
    sensitivitys_val = []
    specificitys_val = []
    accs_val = []

    for i, (train_idx, valid_idx) in enumerate(folds):
        
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        random.seed( 1000 )
        loss = dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_,Xi_test, Xv_test,y_test,early_stopping=False, refit=False)
        #print(len(loss))
        loss_batch.append(loss[-1])#600 is epoch-1
        random.seed( 1000 )
        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        random.seed( 1000 )
        y_test_meta_ = dfm.predict(Xi_test, Xv_test)
        y_test_meta[:,0] += y_test_meta_
        random.seed( 1000 )
        #val
        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx,0])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result
        
        fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
        tprs_val.append(interp(mean_fpr_val, fpr, tpr))
        tprs_val[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs_val.append(roc_auc)
        
        random.seed( 1000 )
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
        cm = confusion_matrix(y_valid_, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = sklearn.metrics.accuracy_score(y_valid_, pre)
        specificitys_val.append(specificity)
        sensitivitys_val.append(sensitivity)
        accs_val.append(acc)
        
        #test
        gini_results_cv_test[i] = gini_norm(y_test, y_test_meta_)
        gini_results_epoch_valid[i] = dfm.valid_result
        gini_results_epoch_test[i] = dfm.test_result 
        
        fpr, tpr, thresholds = roc_curve(y_test, y_test_meta_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        random.seed( 1000 )
        threshold = thresholds[np.argmax(tpr - fpr)]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        cm = confusion_matrix(y_test, pre, labels=[1, 0])
        sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
        acc = sklearn.metrics.accuracy_score(y_test, pre)
        specificitys.append(specificity) 
        sensitivitys.append(sensitivity)
        accs.append(acc)
        
    #val    
    mean_tpr_val = np.mean(tprs_val, axis=0)
    mean_tpr_val[-1] = 1.0
    #mean_auc_val = auc(mean_fpr_val, mean_tpr_val)
    mean_auc_val = np.mean(aucs_val)
    std_auc_val = np.std(aucs_val)
    mean_sensitivity_val = np.mean(sensitivitys_val)
    std_sensitivity_val = np.std(sensitivitys_val)
    mean_specificity_val = np.mean(specificitys_val)
    std_specificity_val = np.std(specificitys_val)
    mean_acc_val = np.mean(accs_val)
    std_acc_val = np.std(accs_val)
    
    ROC_val = sklearn.metrics.roc_auc_score(y_train, y_train_meta)
    
    #test
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_sensitivity = np.mean(sensitivitys)
    std_sensitivity = np.std(sensitivitys)
    mean_specificity = np.mean(specificitys)
    std_specificity = np.std(specificitys)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    y_test_meta /= float(len(folds))
    y_test_meta = np.array(y_test_meta)[:,0]
    #cm
    fpr, tpr, thresholds = roc_curve(y_test, y_test_meta)
    threshold = thresholds[np.argmax(tpr - fpr)]
    pre = (np.array(y_test_meta) >= threshold) * 1
    cm = confusion_matrix(y_test, pre, labels=[1, 0])
    sensitivity = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] * 1.0 / (cm[1, 1] + cm[1, 0])
    acc = sklearn.metrics.accuracy_score(y_test, pre)
    ROC1 = sklearn.metrics.roc_auc_score(y_test, y_test_meta)
    ROC2 = sklearn.metrics.roc_auc_score(y_test, pre)
    #pd.DataFrame({"ID": ids_test, "target": pre.flatten()}).to_csv(os.path.join(config.SUB_DIR, "pre_threshold.csv"), index=False, float_format="%.5f")
    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "deepFM %s"%(config.name)
    elif dfm_params["use_fm"]:
        clf_str = "1126_nolasso_xgboost_FM"
    elif dfm_params["use_deep"]:
        clf_str = "1126_nolasso_xgboost_DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv_test.mean(), gini_results_cv_test.std())
    _make_submission(ids_test, y_test_meta, filename)
    

    _plot_fig(gini_results_epoch_train,gini_results_epoch_valid, gini_results_epoch_test, filename)
    
    _plot_tsne(X_test,y_test,filename)

    return cm, sensitivity,specificity,acc,ROC_val,ROC1,ROC2,loss_batch, y_train_meta, y_test_meta, mean_auc_val,std_auc_val,mean_sensitivity_val,std_sensitivity_val,mean_specificity_val,std_specificity_val,mean_acc_val,std_acc_val, mean_auc,std_auc,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc

def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"ID": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, test_results,model_name):
    colors = ["g", "b", "darkorange","c", "m", "grey","k", "pink", "y","r"]
    step = 1
    xs = np.arange(1, train_results.shape[1]+1,step)
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
#        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
#        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="+")
#        plt.plot(xs, test_results[i], color=colors[i], linestyle="dotted", marker=".")
        plt.plot(xs, np.linspace(train_results[i][0],train_results[i][dfm_params['epoch']-1],dfm_params['epoch']/step), color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, np.linspace(valid_results[i][0],valid_results[i][dfm_params['epoch']-1],dfm_params['epoch']/step), color=colors[i], linestyle="dashed", marker="+")
        plt.plot(xs, np.linspace(test_results[i][0],test_results[i][dfm_params['epoch']-1],dfm_params['epoch']/step), color=colors[i], linestyle="dotted", marker=".")
        legends.append("training set %d"%(i+1))
        legends.append("valid set %d"%(i+1))
        legends.append("test set %d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends,loc="lower right")
    plt.savefig("%s/Plot_%s.pdf"%(config.SUB_DIR,model_name))
    plt.close()

#Visualizing the Training Set
def _plot_tsne(X_train,y_train,model_name):
    tsne = TSNE(n_components=2, random_state=0) #number components
    X_tsne = tsne.fit_transform(X_train)
    
    plt.figure(figsize=(10, 8))
    mask_0 = (y_train == 0)
    mask_1 = (y_train == 1)
    
    plt.scatter(X_tsne[mask_0, 1], X_tsne[mask_0, 0], marker='s', c='g', label='No-HF', edgecolor='k', alpha=0.7)
    plt.scatter(X_tsne[mask_1, 1], X_tsne[mask_1, 0], marker='o', c='r', label='HF', edgecolor='k', alpha=0.7)
    
    
    plt.title('t-SNE plot of the testing data')
    plt.xlabel('1st embedding axis')
    plt.ylabel('2nd embedding axis')
    plt.legend(loc='best', frameon=True, shadow=True)
    
    plt.tight_layout()
    #plt.show()
#    model_name = 20
    plt.savefig("%s/Tsne_%s.pdf"%(config.SUB_DIR,model_name))
    plt.close()

## In[*]
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,#
    "dropout_fm":[1.0,1.0], 
    "deep_layers":[256,256],
    "dropout_deep":[0.6,0.6,0.6],
    "deep_layer_activation":tf.nn.relu,
    "epoch":400,#
    "batch_size":300, 
    "learning_rate":0.0001,#
    "optimizer":"adam",
    "batch_norm":0.5,
    "batch_norm_decay":0.9,
    "l2_reg":0.0001,
    "verbose":True,
    "eval_metric":gini_norm,
    "random_seed":config.RANDOM_SEED
}

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices,ex_list,cols = load_data()

# folds

from sklearn.utils import resample
y =  np.arange(len(y_train))
#config.NUM_SPLITS = 10
#train_index = []
#test_index = []
folds = []
for i in range(config.NUM_SPLITS):
    train = resample(y,n_samples=len(y),replace=1,random_state=1000)
    #train_index.append(train)
    #print(train)
    test = np.array(list(set(y).difference(set(train))))
    #print(test)
    #test_index.append(test)
    folds.append(np.array([train,test]).flatten())
#folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True, random_state=config.RANDOM_SEED).split(X_train, y_train))
## In[*]
#y_train_dfm,y_test_dfm = run_base_model_dfm(dfTrain,dfTest,folds,dfm_params)
#random.seed( 1000 )
cm, sensitivity,specificity,acc,ROC_val,ROC1,ROC2, loss_batch, y_train_meta, y_test_meta, mean_auc_val,std_auc_val,mean_sensitivity_val,std_sensitivity_val,mean_specificity_val,std_specificity_val,mean_acc_val,std_acc_val, mean_auc,std_auc,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc = run_base_model_dfm(dfTrain, dfTest, folds, dfm_params,ex_list)
# In[*]
#save
f = open("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\mirna\\val.txt",'a') 
f.write("mean_auc,std_auc,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc")
f.write("\n")
f.write(str(mean_auc_val))
f.write(" ")
f.write(str(std_auc_val))
f.write(" ")
f.write(str(mean_sensitivity_val))
f.write(" ")
f.write(str(std_sensitivity_val))
f.write(" ")
f.write(str(mean_specificity_val))
f.write(" ")
f.write(str(std_specificity_val))
f.write(" ")
f.write(str(mean_acc_val))
f.write(" ")
f.write(str(std_acc_val))
f.write("\n")
f.close()

f = open("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\mirna\\test.txt",'a') 
f.write("mean_auc,std_auc,sensitivity,mean_sensitivity,std_sensitivity,specificity,mean_specificity,std_specificity,acc,mean_acc,std_acc")
f.write("\n")
f.write(str(mean_auc))
f.write(" ")
f.write(str(std_auc))
f.write(" ")
f.write(str(sensitivity))
f.write(" ")
f.write(str(mean_sensitivity))
f.write(" ")
f.write(str(std_sensitivity))
f.write(" ")
f.write(str(specificity))
f.write(" ")
f.write(str(mean_specificity))
f.write(" ")
f.write(str(std_specificity))
f.write(" ")
f.write(str(acc))
f.write(" ")
f.write(str(mean_acc))
f.write(" ")
f.write(str(std_acc))
f.write(" ")
f.write(str(ROC1))
f.write(" ")
f.write(str(ROC2))
f.write("\n")
f.write(str(cm))
f.write("\n")
f.close()

# In[*]
#调参使用
roc_train = []
sensitivity_train = []
specificity_train = []
acc_train = []

roc_test = []
sensitivity_test = []
specificity_test = []
acc_test = []

loss_score = []

# In[*]
loss_score.append(loss_batch)#3-cv
#train
roc_train.append(mean_auc_val)
sensitivity_train.append(mean_sensitivity_val)
specificity_train.append(mean_specificity_val)
acc_train.append(mean_acc_val)

#test
roc_test.append(mean_auc)
sensitivity_test.append(mean_sensitivity)
specificity_test.append(mean_specificity)
acc_test.append(mean_acc)
# In[*]
param_range1 = np.array([2,4,8,16])
na = "Embedding size"
li = "D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\"
plt.plot(param_range1, roc_train, 'o-', color="g", label="Train:ROC", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, roc_test, 'o-', color="g", linestyle='dashed', label="Test:ROC", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, sensitivity_train, 'o-', color="darkorange", label="Train:Accuracy", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, sensitivity_test, 'o-', color="darkorange", linestyle='dashed', label="Test:Accuracy", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, specificity_train, 'o-', color="b", label="Train:F1 score", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, specificity_test, 'o-', color="b", linestyle='dashed', label="Test:F1 score", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, acc_train, 'o-', color="pink", label="Train:Average Precision", lw=1, alpha=.8,markersize=2)
plt.plot(param_range1, acc_test, 'o-', color="pink", linestyle='dashed', label="Test:Average Precision", lw=1, alpha=.8,markersize=2)
plt.xlabel(na)
plt.ylabel("Value")
plt.legend(fontsize = "x-small",loc="lower center",ncol=2)
plt.savefig('%s%s.pdf'%(li,na))
plt.close()

plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,:1], 'o-', color="g", lw=1, alpha=.8,markersize=2,label="Fold1")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,1:2], 'o-', color="b", lw=1, alpha=.8,markersize=2,label="Fold2")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,2:3], 'o-', color="darkorange", lw=1, alpha=.8,markersize=2,label="Fold3")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,3:4], 'o-', color="c", lw=1, alpha=.8,markersize=2,label="Fold4")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,4:5], 'o-', color="m", lw=1, alpha=.8,markersize=2,label="Fold5")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,5:6], 'o-', color="grey", lw=1, alpha=.8,markersize=2,label="Fold6")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,6:7], 'o-', color="k", lw=1, alpha=.8,markersize=2,label="Fold7")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,7:8], 'o-', color="pink", lw=1, alpha=.8,markersize=2,label="Fold8")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,8:9], 'o-', color="y", lw=1, alpha=.8,markersize=2,label="Fold9")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,9:10], 'o-', color="r", lw=1, alpha=.8,markersize=2,label="Fold10")
plt.xlabel(na)
plt.ylabel("Loss")
plt.legend(fontsize = "x-small",loc=1)
plt.savefig('%s%s_Loss.pdf'%(li,na))
plt.close()
# In[*]
# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm, y_test_fm = run_base_model_dfm(dfTrain, dfTest, folds, fm_params)

# In[*]
# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn, y_test_dnn = run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)