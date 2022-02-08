import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt
import config_20201126 as config
from metrics import gini_norm
from DeepFM import DeepFM


def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    cols = [c for c in dfTrain.columns if c not in ['ID','target']]    
    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values
    X_test = dfTest[cols].values
    ids_test = dfTest['ID'].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices

def run_base_model_dfm(dfTrain,dfTest,folds,dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain,dfTest=dfTest, numeric_cols=config.NUMERIC_COLS,ignore_cols = config.IGNORE_COLS)
    data_parser = DataParser(feat_dict= fd)
    # Xi_train ：列的序号; Xv_train ：列的对应的值
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_test,Xv_test,ids_test = data_parser.parse(df=dfTest)

    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_test_meta = np.zeros((dfTest.shape[0],1),dtype=float)
    _get = lambda x,l:[x[i] for i in l]
    random.seed( 1000 )
    Thresholds = [0.2782, 0.1724, 0.2774, 0.2882, 0.1556, 0.2249, 0.2761,0.3406, 0.2603,0.3164]
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        #如果想要early_stopping和refit，想要改code
        random.seed( 1000 )
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_,Xi_test, Xv_test,y_test=None,early_stopping=False, refit=False)
        random.seed( 1000 )
        y_test_meta_ = dfm.predict(Xi_test, Xv_test)
        y_test_meta[:,0] += y_test_meta_
        random.seed( 1000 )         
        
        #test        
        threshold = Thresholds[i]
        pre = (np.array(y_test_meta_) >= threshold) * 1
        
    #test
    y_test_meta /= float(len(folds))
    y_test_meta = np.array(y_test_meta)[:,0]
    Threshold = 0.2940
    pre = (np.array(y_test_meta) >= Threshold) * 1
    
    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "deepFM %s"%(config.name)
    elif dfm_params["use_fm"]:
        clf_str = "1126_nolasso_xgboost_FM"
    elif dfm_params["use_deep"]:
        clf_str = "1126_nolasso_xgboost_DNN"
    filename = "%s.csv"%clf_str
    _make_submission(ids_test, y_test_meta, pre, filename)
    #测试集的结果，测试集为名称

    return pre, y_test_meta

def _make_submission(ids, y_pred, pre, filename="submission.csv"):
    pd.DataFrame({"ID": ids, "target": y_pred.flatten(), "target_": pre.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


## In[*]
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,#
    "dropout_fm":[1.0,1.0], #无意义参数
    "deep_layers":[256,256],
    "dropout_deep":[0.6,0.6,0.6],
    "deep_layer_activation":tf.nn.relu,
    "epoch":400,#
    "batch_size":300, #几个样本为一组
    "learning_rate":0.0001,#
    "optimizer":"adam",
    "batch_norm":0.5,#无意义参数
    "batch_norm_decay":0.9,
    "l2_reg":0.0001,
    "verbose":True,
    "eval_metric":gini_norm,
    "random_seed":config.RANDOM_SEED
}

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

# folds
from sklearn.utils import resample
y =  np.arange(len(y_train))
folds = []
for i in range(config.NUM_SPLITS):
    train = resample(y,n_samples=len(y),replace=1,random_state=1000)
    #train_index.append(train)
    #print(train)
    test = np.array(list(set(y).difference(set(train))))
    #print(test)
    #test_index.append(test)
    folds.append(np.array([train,test]).flatten())

## In[*]
# ------------------ deepFM Model ------------------
pre, y_test_meta = run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)
