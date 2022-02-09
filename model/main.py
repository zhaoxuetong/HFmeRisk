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
import config as config
from metrics import gini_norm
from DeepFM import DeepFM
from numpy.random import seed
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score
#seed(2020)
from tensorflow import set_random_seed
#set_random_seed(2020) 


## In[*]
def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    ex_list = list(dfTest.ID)
    dfTest = dfTest[dfTest.ID.isin(ex_list)]

# =============================================================================
#     def preprocess(df):
#         cols = [c for c in df.columns if c not in ['id','target']]
#         #df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
#         #df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
#         #df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
#         return df
# 
#     dfTrain = preprocess(dfTrain)
#     dfTest = preprocess(dfTest)
# =============================================================================

    cols = [c for c in dfTrain.columns if c not in ['ID','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['ID'].values

    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices,ex_list

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


    y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
    #y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
    y_test = y_test[y_test.ID.isin(ex_list)]
    #y_test = y_test['JHU_DMP_new.chf'].values.tolist()
    y_test = np.array(y_test)[:,0]
    
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
    
    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "deepFM 25CpG"
    elif dfm_params["use_fm"]:
        clf_str = "1126_nolasso_xgboost_FM"
    elif dfm_params["use_deep"]:
        clf_str = "1126_nolasso_xgboost_DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv_test.mean(), gini_results_cv_test.std())
    _make_submission(ids_test, y_test_meta, filename)
    

    _plot_fig(gini_results_epoch_train,gini_results_epoch_valid, gini_results_epoch_test, clf_str)
    

    return cm, sensitivity,specificity,acc,ROC_val,ROC1,ROC2,loss_batch, y_train_meta, y_test_meta, mean_auc_val,std_auc_val,mean_sensitivity_val,std_sensitivity_val,mean_specificity_val,std_specificity_val,mean_acc_val,std_acc_val, mean_auc,std_auc,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc

def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"ID": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, test_results,model_name):
    colors = ["red", "blue", "yellow"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="+")
        plt.plot(xs, test_results[i], color=colors[i], linestyle="dotted", marker=".")
        legends.append("training set %d"%(i+1))
        legends.append("valid set %d"%(i+1))
        legends.append("test set %d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends,loc="lower right")
    #plt.legend(loc="best",fontsize = "x-small")
    plt.savefig("fig/%s.pdf"%model_name)
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
    "learning_rate":0.0001,
    "optimizer":"adam",
    "batch_norm":0.5,
    "batch_norm_decay":0.9,
    "l2_reg":0.0001,
    "verbose":True,
    "eval_metric":gini_norm,
    "random_seed":config.RANDOM_SEED
}

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices,ex_list = load_data()

# folds
#random.seed( 1000 )
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True, random_state=config.RANDOM_SEED).split(X_train, y_train))
## In[*]
#y_train_dfm,y_test_dfm = run_base_model_dfm(dfTrain,dfTest,folds,dfm_params)
#random.seed( 1000 )
cm, sensitivity,specificity,acc,ROC_val,ROC1,ROC2, loss_batch, y_train_meta, y_test_meta, mean_auc_val,std_auc_val,mean_sensitivity_val,std_sensitivity_val,mean_specificity_val,std_specificity_val,mean_acc_val,std_acc_val, mean_auc,std_auc,mean_sensitivity,std_sensitivity,mean_specificity,std_specificity,mean_acc,std_acc = run_base_model_dfm(dfTrain, dfTest, folds, dfm_params,ex_list)
# In[*]
#save
f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\1126_val.txt",'a') 
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

f = open("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\1126_test.txt",'a') 
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
f.write("\n")
f.write(str(cm))
f.write("\n")
f.close()

# In[*]

roc_score_train = []
acc_score_train = []
f1_score_train = []
AP_score_train = []
roc_score_test = []
acc_score_test = []
f1_score_test = []
AP_score_test = []

loss_score = []

# In[*]
loss_score.append(loss_batch)#3-cv
import sklearn
#train
y_train_meta = y_train_meta[:,0]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_train, y_train_meta, pos_label=1)
roc = auc(fpr, tpr)
threshold = thresholds[np.argmax(tpr - fpr)]
pre = (np.array(y_train_meta) >= threshold) * 1
average_precision = sklearn.metrics.average_precision_score(y_train, pre, pos_label=1)

roc_score_train.append(sklearn.metrics.roc_auc_score(y_train, y_train_meta))
acc_score_train.append(sklearn.metrics.accuracy_score(y_train, pre))
f1_score_train.append(sklearn.metrics.fbeta_score(y_train, pre, beta=1))
AP_score_train.append(average_precision)
#test
y_test = pd.read_table("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU.txt",sep=' ')
y_test = y_test[y_test.ID.isin(ex_list)]
#y_test = y_test['JHU_DMP_new.chf'].values.tolist()
y_test = np.array(y_test)[:,0]
        
y_test_meta = y_test_meta[:,0]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_test_meta, pos_label=1)
roc = auc(fpr, tpr)
threshold = thresholds[np.argmax(tpr - fpr)]
pre = (np.array(y_test_meta) >= threshold) * 1
average_precision = sklearn.metrics.average_precision_score(y_test, pre, pos_label=1)

roc_score_test.append(sklearn.metrics.roc_auc_score(y_test, y_test_meta))
acc_score_test.append(sklearn.metrics.accuracy_score(y_test, pre))
f1_score_test.append(sklearn.metrics.fbeta_score(y_test, pre, beta=1))
AP_score_test.append(average_precision)
# In[*]
param_range1 = np.array([0.4,0.5,0.6,0.7,0.8,0.9])
plt.plot(param_range1, roc_score_train, 'o-', color="r", label="Training")
plt.plot(param_range1, roc_score_test, 'o-', color="g", label="Test")
plt.xlabel("dropout_deep")
plt.ylabel("roc")
plt.legend(loc="upper left")
plt.savefig('./dropout_deep roc.pdf')
#plt.show()
plt.plot(param_range1, acc_score_train, 'o-', color="r", label="Training")
plt.plot(param_range1, acc_score_test, 'o-', color="g", label="Test")
plt.xlabel("dropout_deep")
plt.ylabel("acc")
plt.legend(loc="upper left")
plt.savefig('./dropout_deep acc.pdf')
plt.plot(param_range1, f1_score_train, 'o-', color="r", label="Training")
plt.plot(param_range1, f1_score_test, 'o-', color="g", label="Test")
plt.xlabel("dropout_deep")
plt.ylabel("f1")
plt.legend(loc="upper left")
plt.savefig('./dropout_deep f1.pdf')
plt.plot(param_range1, AP_score_train, 'o-', color="r", label="Training")
plt.plot(param_range1, AP_score_test, 'o-', color="g", label="Test")
plt.xlabel("dropout_deep")
plt.ylabel("AP")
plt.legend(loc="upper left")
plt.savefig('./dropout_deep AP.pdf')

plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,:1], 'o-', color="r", label="Training1")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,1:2], 'o-', color="b", label="Training2")
plt.plot(param_range1, np.array(loss_score)[:len(param_range1)+1,2:3], 'o-', color="g", label="Training3")
plt.xlabel("dropout_deep")
plt.ylabel("loss")
plt.legend(loc="upper left")
plt.savefig('./dropout_deep loss.pdf')
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