#train for Adversarial validation
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  {
    load(file="train_meta_raw.Rdata")
    train_meta = train_meta_raw[,!colnames(train_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                   "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                   "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                   "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                   "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                   "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                   "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                   "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
    train_meta = train_meta[,-c(1,2)]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Numberofcigarettessmoked")]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Heartfailure")]
    
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Diabetes")]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Rightventricularhypertrophy")]
    library(tibble)
    library(impute)
    Patient_impute <- impute.knn(as.matrix(data.frame(t(train_meta))))
    Patient_impute <- data.frame(t(Patient_impute$data))
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement","Leftventricularhypertrophy",
               "Neurological","Gidisease","Infectious",
               "Fever","Chronicbronchitis","COPD")){
      Patient_impute[,i] <- round(Patient_impute[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein","Hight")){
      Patient_impute[,i] <- round(Patient_impute[,i],2)
    }
    Patient_impute <- Patient_impute[,-c(88:93)]
  }
  X <- data.frame(Patient_impute)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation1.csv",row.names = F,sep=",")
}
#test for Adversarial validation
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  {
    load(file="test/test_meta_raw.Rdata")
    test_meta = test_meta_raw[,!colnames(test_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                   "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                   "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                   "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                   "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                   "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                   "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                   "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
    test_meta = test_meta[,-c(1,2)]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Numberofcigarettessmoked")]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Heartfailure")]
    
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Diabetes")]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Rightventricularhypertrophy")]
    library(tibble)
    library(impute)
    Patient_impute <- impute.knn(as.matrix(data.frame(t(test_meta))))
    Patient_impute <- data.frame(t(Patient_impute$data))
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement","Leftventricularhypertrophy",
               "Neurological","Gidisease","Infectious",
               "Fever","Chronicbronchitis","COPD",
               "Treatedforlipids","Drinkwine")){
      Patient_impute[,i] <- round(Patient_impute[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein","Hight")){
      Patient_impute[,i] <- round(Patient_impute[,i],2)
    }
    Patient_impute <- Patient_impute[,-c(88:93)]
  }
  X <- data.frame(Patient_impute)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation1_test.csv",row.names = F,sep=",")
}

#train
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary")
  load("tmp_train.Rdata")#90 ehr impute after
  data = Patient_impute[,-c(1:3)]
  X <- data.frame(data)
  X = X[,-c(92:97)]#"CD8T"  "CD4T"  "NK"    "Bcell" "Mono"  "Gran" 
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation2.csv",row.names = F,sep=",")
}
#test
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary")
  load("tmp_test.Rdata")
  data = Patient_impute_test[,-c(1:3)]
  X <- data.frame(data)
  X = X[,-c(92:97)]#"CD8T"  "CD4T"  "NK"    "Bcell" "Mono"  "Gran" 
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation2_test.csv",row.names = F,sep=",")
}

#train
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary")
  load("tmp_train.Rdata")#90 ehr impute after
  data = Patient_impute[,-c(1:3)]
  X <- data.frame(data)
  X = X[,-c(92:97)]#"CD8T"  "CD4T"  "NK"    "Bcell" "Mono"  "Gran" 
  set.seed(1234)
  samp=sample(1:nrow(X),171)	
  X=X[samp,]
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation3.csv",row.names = F,sep=",")
}
#test
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary")
  load("tmp_test.Rdata")
  data = Patient_impute_test[,-c(1:3)]
  X <- data.frame(data)
  X = X[,-c(92:97)]#"CD8T"  "CD4T"  "NK"    "Bcell" "Mono"  "Gran" 
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation3_test.csv",row.names = F,sep=",")
}

#train for Adversarial validation
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  {
    load(file="train_meta_raw.Rdata")
    train_meta = train_meta_raw[,!colnames(train_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                   "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                   "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                   "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                   "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                   "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                   "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                   "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
    train_meta = train_meta[,-c(1,2)]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Numberofcigarettessmoked")]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Heartfailure")]
    
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Diabetes")]
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Rightventricularhypertrophy")]
    library(tibble)
    library(impute)
    Patient_impute <- impute.knn(as.matrix(data.frame(t(train_meta))))
    Patient_impute <- data.frame(t(Patient_impute$data))
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement","Leftventricularhypertrophy",
               "Neurological","Gidisease","Infectious",
               "Fever","Chronicbronchitis","COPD")){
      Patient_impute[,i] <- round(Patient_impute[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein","Hight")){
      Patient_impute[,i] <- round(Patient_impute[,i],2)
    }
    Patient_impute <- Patient_impute[,-c(88:93)]
  }
  X <- data.frame(Patient_impute)
  set.seed(1234)
  samp=sample(1:nrow(X),171)	
  X=X[samp,]
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation4.csv",row.names = F,sep=",")
}
#test for Adversarial validation
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  {
    load(file="test/test_meta_raw.Rdata")
    test_meta = test_meta_raw[,!colnames(test_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
    test_meta = test_meta[,-c(1,2)]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Numberofcigarettessmoked")]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Heartfailure")]
    
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Diabetes")]
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Rightventricularhypertrophy")]
    library(tibble)
    library(impute)
    Patient_impute <- impute.knn(as.matrix(data.frame(t(test_meta))))
    Patient_impute <- data.frame(t(Patient_impute$data))
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement","Leftventricularhypertrophy",
               "Neurological","Gidisease","Infectious",
               "Fever","Chronicbronchitis","COPD",
               "Treatedforlipids","Drinkwine")){
      Patient_impute[,i] <- round(Patient_impute[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein","Hight")){
      Patient_impute[,i] <- round(Patient_impute[,i],2)
    }
    Patient_impute <- Patient_impute[,-c(88:93)]
  }
  X <- data.frame(Patient_impute)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20210819deepfm_feature_AdversarialValidation4_test.csv",row.names = F,sep=",")
}
