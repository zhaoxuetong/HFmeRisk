# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:07:40 2021

@author: zhaoxt
"""
#https://www.cnblogs.com/MiQing4in/p/13397596.html
#https://zhuanlan.zhihu.com/p/205183444

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC
from sklearn.metrics import auc
import random

name = "AdversarialValidation1"#dmp_lassoxgboost;age1Coefficient5EHR
names= ""
TRAIN_FILE = "data\\20210819deepfm_feature_%s%s.csv"%(name,names)
TEST_FILE = "data\\20210819deepfm_feature_%s%s_test.csv"%(name,names)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
cols = [c for c in train.columns if c not in ['ID','target']]#'Diuretic',"Age","BMI","Gender","Heartfailure"
x_train = train[cols]
x_test = test[cols]
x_train['target']=0# 开始给训练集和测试集打上标签
x_test['target']= 1
data=pd.concat([x_train,x_test],axis=0)#将训练集和测试集合并
train_label=data['target']#取出train_label,方便后面巡礼N
data.drop(['target'],axis=1,inplace=True)
train_data=data

#kf=StratifiedKFold(n_splits=6,shuffle=True,random_state=123)
x,y=pd.DataFrame(train_data),pd.DataFrame(train_label)  #将数据dataframe化，后面进行iloc 等选项

_get = lambda x,l:[x[i] for i in l]
random.seed( 1000 )
folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=100).split(x, y))

for i, (train_idx, valid_idx) in enumerate(folds):
    
    print("第",i+1,"次")
    X_train_, y_train_ = _get(x.values, train_idx), _get(y.values, train_idx)
    X_valid_, y_valid_ = _get(x.values, valid_idx), _get(y.values, valid_idx)
    algorithm = LinearSVC()
    random.seed( 1000 )
    algorithm.fit(X_train_, y_train_)
    y_train_meta = np.zeros((x.shape[0],1),dtype=float)
    random.seed( 1000 )
    y_train_meta[valid_idx,0] = algorithm.decision_function(X_valid_)
    random.seed( 1000 )
    fpr, tpr, thresholds = roc_curve(y_valid_, y_train_meta[valid_idx,0])
    random.seed( 1000 )
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    threshold = thresholds[np.argmax(tpr - fpr)]
    print(threshold)
    random.seed( 1000 )
    pre = (np.array(y_train_meta[valid_idx,0]) >= threshold) * 1
    print(matthews_corrcoef(np.array(y_valid_)[:,0], pre))
    #对测试集进行操作
random.seed( 1000 )
fpr, tpr, thresholds = roc_curve(y, y_train_meta)
random.seed( 1000 )
roc_auc = auc(fpr, tpr)
print(roc_auc)
random.seed( 1000 )
threshold = thresholds[np.argmax(tpr - fpr)]
print(threshold)
pre = (np.array(y_train_meta) >= threshold) * 1
random.seed( 1000 )
print(matthews_corrcoef(np.array(y)[:,0], pre))

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='r',label='SVM model:AUC = %0.3f' % roc_auc,lw=lw, alpha=1.0,markersize=12)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=.8)
plt.xlim([-0.02, 1.05])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.ylabel('True Positive Rate',fontdict={'family':'Times New Roman','size':10})
plt.title('Covariate Shift: Receiver operating characteristic',fontdict={'family':'Times New Roman','size':10})
plt.legend(loc=4,fontsize = "x-small",prop={'family':'Times New Roman','size':8})
plt.savefig("D:\\anaconda-python\\UMN_JHU_alldata\\trainUMN_testJHU\\new_result\\deepfm\\output\\new_1126\\0827Covariate Shift_SVMmodel.pdf")

#对训练集和测试集做的
from scipy import stats
for i in range(train.shape[1]):
    print(cols[i],stats.ks_2samp(train.values[:,i], test.values[:,i]))

#Ejectionfraction Ks_2sampResult(statistic=0.08365434707639026, pvalue=0.2609097363123599)
#Omega3 Ks_2sampResult(statistic=0.06128244073169121, pvalue=0.6390575068630184)
#Statin Ks_2sampResult(statistic=0.3663665646760146, pvalue=7.702447215575547e-17)
#Thiazides Ks_2sampResult(statistic=0.0503936545672001, pvalue=0.8458457594871575)
#Diuretic Ks_2sampResult(statistic=0.04548489584479811, pvalue=0.9173622344243776)
#Potassium Ks_2sampResult(statistic=0.01716231188594657, pvalue=0.9999999999390203)
#Aldosterone Ks_2sampResult(statistic=0.07308840901920212, pvalue=0.4164055906224715)
#Amiodarone Ks_2sampResult(statistic=0.002509410288582183, pvalue=1)
#Vasodilators Ks_2sampResult(statistic=0.07308840901920212, pvalue=0.4164055906224715)
#CoQ10 Ks_2sampResult(statistic=0.005400368340340605, pvalue=1.0)
#Betablocking Ks_2sampResult(statistic=0.28422373373836096, pvalue=1.537433513831843e-10)
#AngiotensinIIantagonists Ks_2sampResult(statistic=0.0354252423195169, pvalue=0.9913388160235281)
#ACEI Ks_2sampResult(statistic=0.2010976835648301, pvalue=1.812635338116486e-05)
#Warfarin Ks_2sampResult(statistic=0.020860390205962417, pvalue=0.9999998447726758)
#Clopidogrel Ks_2sampResult(statistic=0.08144577252415858, pvalue=0.2893322471829566)
#Aspirin Ks_2sampResult(statistic=0.3045044648425749, pvalue=4.637179529254354e-12)
#Folicacid Ks_2sampResult(statistic=0.04254991305113474, pvalue=0.9493437976118437)
#Coronaryheartdisease Ks_2sampResult(statistic=0.2539273738507708, pvalue=1.7458116730040274e-08)
#Heartfailure Ks_2sampResult(statistic=0.11310689941080221, pvalue=0.04982309719818101)
#Myocardialinfarction Ks_2sampResult(statistic=0.11904290211098638, pvalue=0.03359616169888213)
#Diabetes Ks_2sampResult(statistic=0.1596337141473508, pvalue=0.0013210483535550654)
#Atrialfibrillation Ks_2sampResult(statistic=0.06847314857616647, pvalue=0.4987550593777357)
#Stroke Ks_2sampResult(statistic=0.060563369947243685, pvalue=0.6534678500427418)
#Gender Ks_2sampResult(statistic=0.33370020618254126, pvalue=1.84297022087776e-14)
#Age Ks_2sampResult(statistic=0.23963400764563017, pvalue=1.3553634048424357e-07)
#Bloodglucose Ks_2sampResult(statistic=0.3572240932737532, pvalue=4.961710018877332e-16)
#BMI Ks_2sampResult(statistic=0.3935445053453374, pvalue=2.296050582735732e-19)
#LDLcholesterol Ks_2sampResult(statistic=0.11179349461063784, pvalue=0.05421687930314956)
#Numberofcigarettessmoked Ks_2sampResult(statistic=0.037685179070637696, pvalue=0.983110598034503)
#Creatinineserum Ks_2sampResult(statistic=0.3313668948615789, pvalue=2.964295475749168e-14)
#Smoking Ks_2sampResult(statistic=0.041874867008592165, pvalue=0.9556522270110792)
#Averagediastolicbloodpressure Ks_2sampResult(statistic=0.2689251359263906, pvalue=1.8039574278816417e-09)
#Leftventricularhypertrophy Ks_2sampResult(statistic=0.018820577164366373, pvalue=0.9999999964562477)
#Fastingbloodglucose Ks_2sampResult(statistic=0.3576496657788344, pvalue=4.554354248555585e-16)
#HDLcholesterol Ks_2sampResult(statistic=0.13840645109218047, pvalue=0.0080748492287539)
#Hight Ks_2sampResult(statistic=0.20721712268961823, pvalue=8.799157428129867e-06)
#Averagesystolicbloodpressure Ks_2sampResult(statistic=0.04901421265417832, pvalue=0.8682853133177235)
#Totalcholesterol Ks_2sampResult(statistic=0.06500253142265953, pvalue=0.5653457232461865)
#Triglycerides Ks_2sampResult(statistic=0.2722196541122778, pvalue=1.0795365712468197e-09)
#Ventricularrate Ks_2sampResult(statistic=0.09802842530835663, pvalue=0.12378319532780258)
#Waist Ks_2sampResult(statistic=0.26697337236860447, pvalue=2.422331002449596e-09)
#Weight Ks_2sampResult(statistic=0.09239325834452296, pvalue=0.1682000880336827)
#Treatedforhypertension Ks_2sampResult(statistic=0.2730928114933926, pvalue=9.365788145032639e-10)
#Treatedforlipids Ks_2sampResult(statistic=0.3897143527996067, pvalue=5.343842146548376e-19)
#Drinkbeer Ks_2sampResult(statistic=0.004585910615099019, pvalue=1.0)
#Drinkwine Ks_2sampResult(statistic=0.06048265792041794, pvalue=0.6550674876456024)
#Drinkliquor Ks_2sampResult(statistic=0.02085305274897826, pvalue=0.9999998466364483)
#Sleep Ks_2sampResult(statistic=0.8328747422718235, pvalue=2.9538317780452976e-85)
#Albuminurine Ks_2sampResult(statistic=0.2173208009568044, pvalue=2.5643255505691798e-06)
#Creatinineurine Ks_2sampResult(statistic=0.07927388525684768, pvalue=0.3194218512727901)
#HemoglobinA1cwholeblood Ks_2sampResult(statistic=0.17688407551710728, pvalue=0.0002507598683951784)
#Atrialenlargement Ks_2sampResult(statistic=0.031727163999501054, pvalue=0.9979368675275346)
#Rightventricularhypertrophy Ks_2sampResult(statistic=0.030112923462986198, pvalue=0.9990795355209018)
#Rheumatic Ks_2sampResult(statistic=0.002934982793663372, pvalue=1.0)
#Aorticvalve Ks_2sampResult(statistic=0.041720780411924833, pvalue=0.9567642484703144)
#Mitralvalve Ks_2sampResult(statistic=0.04089164777271493, pvalue=0.963473049387389)
#Arrhythmia Ks_2sampResult(statistic=0.03753109247397037, pvalue=0.9837094687308644)
#Dementia Ks_2sampResult(statistic=0.0033385429277920857, pvalue=1.0)
#Parkinson Ks_2sampResult(statistic=0.00878293601003764, pvalue=1.0)
#Adultseizuredisorder Ks_2sampResult(statistic=0.0020838377835009944, pvalue=1)
#Neurological Ks_2sampResult(statistic=0.0226287173391446, pvalue=0.9999981469345283)
#Thyroid Ks_2sampResult(statistic=0.022408593629619847, pvalue=0.9999985882436547)
#Endocrine Ks_2sampResult(statistic=0.01635519161768914, pvalue=0.999999999994527)
#Renal Ks_2sampResult(statistic=0.03125756675251491, pvalue=0.9983454855648579)
#Gynecologic Ks_2sampResult(statistic=0.053974333575469415, pvalue=0.7827843700957668)
#Emphysema Ks_2sampResult(statistic=0.0028909580517584217, pvalue=1)
#Pneumonia Ks_2sampResult(statistic=0.00378612780382575, pvalue=1.0)
#Asthma Ks_2sampResult(statistic=0.028498682926471345, pvalue=0.9996416415491035)
#Pulmonary Ks_2sampResult(statistic=0.035175768782055514, pvalue=0.9920759452269916)
#Gout Ks_2sampResult(statistic=0.03130159149441986, pvalue=0.9983090337765563)
#Degenerative Ks_2sampResult(statistic=0.04640941542480207, pvalue=0.9057835812395012)
#Rheumatoidarthritis Ks_2sampResult(statistic=0.006699098226536647, pvalue=1.0)
#Musculoskeletal Ks_2sampResult(statistic=0.037369668420318886, pvalue=0.9844785486754418)
#Gallbladder Ks_2sampResult(statistic=0.0028909580517584217, pvalue=1)
#Gerd Ks_2sampResult(statistic=0.017793333186584194, pvalue=0.9999999996704556)
#Liver Ks_2sampResult(statistic=0.00963408102020002, pvalue=1.0)
#Gidisease Ks_2sampResult(statistic=0.0437385810825684, pvalue=0.9375106153701078)
#Hematologicdisorder Ks_2sampResult(statistic=0.026708343422336685, pvalue=0.9998961167257331)
#Bleedingdisorder Ks_2sampResult(statistic=0.009186496144166355, pvalue=1.0)
#Eye Ks_2sampResult(statistic=0.015305935268954485, pvalue=0.9999999999998629)
#Ent Ks_2sampResult(statistic=0.02249664311342975, pvalue=0.9999984143318319)
#Skin Ks_2sampResult(statistic=0.06376983864932091, pvalue=0.5895417406394831)
#Depression Ks_2sampResult(statistic=0.023523887091211927, pvalue=0.9999946489887167)
#Anxiety Ks_2sampResult(statistic=0.039769016854138695, pvalue=0.971597846158273)
#Psychosis Ks_2sampResult(statistic=0.0012547051442910915, pvalue=1)
#Prostate Ks_2sampResult(statistic=0.11727457497780419, pvalue=0.03786243088245611)
#Infectious Ks_2sampResult(statistic=0.022225157205015885, pvalue=0.9999988975416942)
#Fever Ks_2sampResult(statistic=0.07724874712921996, pvalue=0.3494029972161632)
#Chronicbronchitis Ks_2sampResult(statistic=0.045594957699560484, pvalue=0.9162528876871665)
#COPD Ks_2sampResult(statistic=0.0113143586695723, pvalue=1.0)
#Creactiveprotein Ks_2sampResult(statistic=0.3299801154915729, pvalue=3.963496197911809e-14)