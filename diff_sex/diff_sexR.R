#diff sex################
#train old:DMP+clin lasso+xgboost30
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  load("UMN_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf","SEX",id3)]
  X = X[,c(2,1,3:32)]
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  colnames(X)[2] <- c("target")#chf
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_pdata_sex_train.txt",row.names = F)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_sex.csv",row.names = F,sep=",")
  # write.table(X[,colnames(X) %in% c("ID","target","AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
  #             "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_ehr_sex.csv",row.names = F,sep=",")
  # write.table(X[,!colnames(X) %in% c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
  #             "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_cpg_sex.csv",row.names = F,sep=",")
}

#test old sex:DMP+clin lasso+xgboost
{
  load(file="JHU_DMP_new.Rdata")
  X <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c("SEX",id3)]
  table(X$SEX)
  # 0   1 
  # 121  50 
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  pdata <- data.frame(JHU_DMP_new$chf)
  pdata$ID <- rownames(JHU_DMP_new)
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_pdata_sex_test.txt",row.names = F)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_sex_test.csv",row.names = F,sep=",")
  write.table(X[,colnames(X) %in% c("ID","AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
              "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_ehr_sex_test.csv",row.names = F,sep=",")
  write.table(X[,!colnames(X) %in% c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
              "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_cpg_sex_test.csv",row.names = F,sep=",")
}

#add sex feature###########
#train old:DMP+clin lasso+xgboost30
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  load("UMN_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf","SEX",id3)]
  X = X[,c(2,1,3:32)]
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  colnames(X)[2] <- c("target")#chf
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_pdata_sex_train.txt",row.names = F)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_sex.csv",row.names = F,sep=",")
  # write.table(X[,colnames(X) %in% c("ID","target","AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
  #             "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_ehr_sex.csv",row.names = F,sep=",")
  # write.table(X[,!colnames(X) %in% c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
  #             "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_cpg_sex.csv",row.names = F,sep=",")
}

#test old sex:DMP+clin lasso+xgboost
{
  load(file="JHU_DMP_new.Rdata")
  X <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c("SEX",id3)]
  table(X$SEX)
  # 0   1 
  # 121  50 
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  pdata <- data.frame(JHU_DMP_new$chf)
  pdata$ID <- rownames(JHU_DMP_new)
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_pdata_sex_test.txt",row.names = F)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_sex_test.csv",row.names = F,sep=",")
  write.table(X[,colnames(X) %in% c("ID","AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
              "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_ehr_sex_test.csv",row.names = F,sep=",")
  write.table(X[,!colnames(X) %in% c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")],
              "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210817deepfm_feature_dmp_lassoxgboost_cpg_sex_test.csv",row.names = F,sep=",")
}
