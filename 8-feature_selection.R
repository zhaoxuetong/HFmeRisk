#train
{
  #DMP+clin lasso+xgboost30
  {
    setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
    load("train_data.Rdata")
    id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
    head(id3)
    id3 <- as.character(id3$Feature)
    X <- train_data[,colnames(train_data) %in% c("Heart.failure",id3)]
    #X <- subset(X, select=c("Heart.failure",id3))
    X = X[,c(2,1,3:31)]
    X <- data.frame(X)
    X <- rownames_to_column(X,"ID")
    #X$ID <-  gsub('X','',X$ID)
    colnames(X)[2] <- c("target")#chf
    write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_pdata_train.txt",row.names = F)
    write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost.csv",row.names = F,sep=",")
    write.table(X[,colnames(X) %in% c("ID","target","Age","Diuretic","BMI","Creatinine.serum","Albumin.urine")],
                "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_ehr.csv",row.names = F,sep=",")
    write.table(X[,!colnames(X) %in% c("Age","Diuretic","BMI","Creatinine.serum","Albumin.urine")],
                "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_cpg.csv",row.names = F,sep=",")
  }
}

#test
{
  #DMP+clin lasso+xgboost
  {
    load("test/test_data.Rdata")
    X <- test_data[,colnames(test_data) %in% c(id3)]
    #X <- subset(X, select=c(id3))
    #X = X[,c(1,9:30,2:8)]
    X <- data.frame(X)
    X <- rownames_to_column(X,"ID")
    #X$ID <-  gsub('X','',X$ID)
    pdata <- data.frame(test_data$Heart.failure)
    pdata$ID <- rownames(test_data)
    #pdata$ID <-  gsub('X','',pdata$ID)
    write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_pdata_test.txt",row.names = F)
    write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_test.csv",row.names = F,sep=",")
    write.table(X[,colnames(X) %in% c("ID","Age","Diuretic","BMI","Creatinine.serum","Albumin.urine")],
                "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_ehr_test.csv",row.names = F,sep=",")
    write.table(X[,!colnames(X) %in% c("Age","Diuretic","BMI","Creatinine.serum","Albumin.urine")],
                "D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_cpg_test.csv",row.names = F,sep=",")
  }
}

############################
#feature 30-1
{
  load("UMN_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  for(i in 3:32){
    X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3)]
    X <- subset(X, select = c("chf",id3))
    X <- rownames_to_column(X,"ID")
    colnames(X)[2] <- "target"
    names = colnames(X)[i]
    X <- X[,-i]
    X <- data.frame(X)
    write.csv(X,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_rm",names,"dmp_lassoxgboost.csv",sep="_"),row.names = F)
  }
  load("JHU_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  for(i in 2:31){
    X <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3)]
    X <- subset(X, select = c(id3))
    X <- rownames_to_column(X,"ID")
    X$ID <-  gsub('X','',X$ID)
    names = colnames(X)[i]
    X <- X[,-i]
    X <- data.frame(X)
    write.csv(X,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_rm",names,"dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
}
#50%,75%,25%,60%
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/DMP_pipeline")
  load("UMN_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3)]
  samp1=sample(1:nrow(X),round(nrow(X)/4))	# 25%
  samp2=sample(1:nrow(X),round(nrow(X)/2))	# 50%
  samp3=sample(1:nrow(X),6*round(nrow(X)/10))	# 60%
  samp4=sample(1:nrow(X),7.5*round(nrow(X)/10))	# 75%
  X=X[samp3,]
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  colnames(X)[3] <- c("target")#chf
  pdata <- data.frame(UMN_DMP_new$chf)
  #write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_UMN.txt",row.names = F)
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_25percentage.csv",row.names = F,sep=",")
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_50percentage.csv",row.names = F,sep=",")
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_60percentage.csv",row.names = F,sep=",")
  write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_75percentage.csv",row.names = F,sep=",")
  
}
#train:sample 6/4-years
{
  library(dplyr)
  library(tibble)
  load(file="UMN_meta_beta.Rdata")
  
  #==========================
  load("UMN_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3)]
  X <- subset(X, select = c("chf",id3))
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  X$ID <-  gsub('X','',X$ID)
  colnames(X)[2] <- "target"
  X <- merge(UMN_meta_beta[,c("shareid","DATE8","chfdate")],X,by.x = "shareid",by.y="ID")
  #================
  #6 years-777
  X_control <- filter(X, target == 0 ,(chfdate - DATE8 ) > 6*365) 
  X_chf <- filter(X, target == 1 & (chfdate >= DATE8) & (chfdate - DATE8 ) <6*365)
  summary((X_control$chfdate - X_control$DATE8))
  summary((X_chf$chfdate - X_chf$DATE8))
  X <- rbind(X_chf,X_control) 
  X <- X[,-c(2,3)] 
  colnames(X)[1] <- "ID"
  samp=sample(1:nrow(X),round(nrow(X)))
  X = X[samp,]
  write.csv(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_6years.csv",row.names = F)
  #================
  #4 years-761
  X_control <- filter(X, target == 0 ,(chfdate - DATE8 ) > 4*365) 
  X_chf <- filter(X, target == 1 & (chfdate >= DATE8) & (chfdate - DATE8 ) <4*365) 
  summary((X_control$chfdate - X_control$DATE8))
  summary((X_chf$chfdate - X_chf$DATE8))
  X <- rbind(X_chf,X_control) 
  X <- X[,-c(2,3)] 
  colnames(X)[1] <- "ID"
  samp=sample(1:nrow(X),round(nrow(X)))
  X = X[samp,]
  write.csv(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_4years.csv",row.names = F)
  
}
#test:sample 6/4-years
{
  library(dplyr)
  library(tibble)
  load(file="test/JHU_meta_beta.Rdata")
  #==========================
  load("test/JHU_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  X <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c("chf",id3)]
  X <- subset(X, select = c("chf",id3))
  X <- data.frame(X)
  X <- rownames_to_column(X,"ID")
  X$ID <-  gsub('X','',X$ID)
  colnames(X)[2] <- "target"
  X <- merge(JHU_meta_beta[,c("shareid","DATE8","chfdate")],X,by.x = "shareid",by.y="ID")
  #================
  #6 years-163
  X_control <- filter(X, target == 0 ,(chfdate - DATE8 ) > 6*365) 
  X_chf <- filter(X, target == 1 & (chfdate >= DATE8) & (chfdate - DATE8 ) <6*365)
  summary((X_control$chfdate - X_control$DATE8))
  summary((X_chf$chfdate - X_chf$DATE8))
  X <- rbind(X_chf,X_control) 
  X <- X[,-c(2,3)] 
  colnames(X)[1] <- "ID"
  samp=sample(1:nrow(X),round(nrow(X)))
  X = X[samp,]
  pdata <- data.frame(X$target)
  pdata$ID <- X$ID
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_6years.txt",row.names = F)#为剔除样本做准备
  X <- X[,-2]
  write.csv(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_6years_test.csv",row.names = F)
  #================
  #4 years-154
  X_control <- filter(X, target == 0 ,(chfdate - DATE8 ) > 4*365) 
  X_chf <- filter(X, target == 1 & (chfdate >= DATE8) & (chfdate - DATE8 ) <4*365)
  summary((X_control$chfdate - X_control$DATE8))
  summary((X_chf$chfdate - X_chf$DATE8))
  X <- rbind(X_chf,X_control) 
  
  X <- X[,-c(2,3)] 
  colnames(X)[1] <- "ID"
  samp=sample(1:nrow(X),round(nrow(X)))
  X = X[samp,]
  pdata <- data.frame(X$target)
  pdata$ID <- X$ID
  write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_4years.txt",row.names = F)#为剔除样本做准备
  X <- X[,-2]
  write.csv(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_dmp_lassoxgboost_4years_test.csv",row.names = F)
  
}
#FHS model : test data
{
  library("readxl")
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  echo1 <- read_excel("phs000007.v30.pht002572.v6.p11.c1.t_echo_2008_m_0549s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  echo1 <- data.frame(echo1)
  setwd("H:/dbgap_CHD/RootStudyConsentSet_phs000007.Framingham.v30.p11.c2.HMB-IRB-NPU-MDS/PhenotypeFiles")
  echo2 <- read.table("phs000007.v30.pht002572.v6.p11.c2.t_echo_2008_m_0549s.HMB-IRB-NPU-MDS.txt",sep="\t",header = T)
  echo2 <- data.frame(echo2)
  echo <- rbind(echo1,echo2)
  echo <- echo[,c("shareid","x139","VRNORMAL","sdate")]
  colnames(echo)[2] <- c("heart_rate")
  colnames(echo)[3] <- c("VALVE_disease")
  colnames(echo)[4] <- c("echo_date")
  echo <- filter(echo,VALVE_disease != "" )
  echo$VALVE_disease <- ifelse(echo$VALVE_disease == "2" ,"1","0")
  
  #LVH
  setwd("H:/dbgap_CHD/RootStudyConsentSet_phs000007.Framingham.v30.p11.c1.HMB-IRB-MDS/PhenotypeFiles")
  LVH1 <- read.table("phs000007.v30.pht000747.v6.p11.c1.ex1_8s.HMB-IRB-MDS.txt",sep="\t",header = T,skip=10)
  LVH1 <- data.frame(LVH1)
  setwd("H:/dbgap_CHD/RootStudyConsentSet_phs000007.Framingham.v30.p11.c2.HMB-IRB-NPU-MDS/PhenotypeFiles")
  LVH2 <- read.table("phs000007.v30.pht000747.v6.p11.c2.ex1_8s.HMB-IRB-NPU-MDS.txt",sep="\t",header = T,skip=10)
  LVH2 <- data.frame(LVH2)
  LVH <- rbind(LVH1,LVH2)
  LVH <- LVH[,c("shareid","H338")]
  colnames(LVH)[2] <- c("LVH") 
  LVH <- filter(LVH,LVH != "" )
  LVH$LVH <- ifelse(LVH$LVH == "0" ,"0","1")
  FHS <- merge(echo,LVH,by="shareid")
  
  load("test/JHU_meta_beta.Rdata")
  tmp_test = JHU_meta_beta[,colnames(JHU_meta_beta) %in% 
                            c("shareid","chf","SEX",
                              "SBP8","CURR_DIAB8",
                              "BMI8","AGE8","chd",id3)]
  tmp_test = merge(FHS,tmp_test,by="shareid")
  #add col
  tmp_test <- filter(tmp_test,CURR_DIAB8 != "" )
  tmp_test$vd_diab <- ifelse(tmp_test$CURR_DIAB8 == "1" & tmp_test$VALVE_disease == "1","1","0") #且
  tmp_test <- column_to_rownames(tmp_test,"shareid")
  #impute
  tmp_test <- impute.knn(as.matrix(data.frame(t(tmp_test))))
  tmp_test <- data.frame(t(tmp_test$data))
  #change 0,1,
  for(i in c("heart_rate","BMI8","Albumin_urine")){
    tmp_test[,i] <- round(tmp_test[,i],2)
  }
  #rebulit data
  {
    FHS_man <- tmp_test[tmp_test$SEX == 0, colnames(tmp_test) %in% c('chd','heart_rate','VALVE_disease','LVH','AGE8','SBP8','CURR_DIAB8')]
    FHS_woman <- tmp_test[tmp_test$SEX == 1,colnames(tmp_test) %in% c('chd','heart_rate','VALVE_disease','LVH','AGE8','SBP8','CURR_DIAB8','vd_diab','BMI8')]
    
    normalization<-function(x){
      return((x-min(x))/(max(x)-min(x)))}
    #c= c("AGE8","BMI8","CREAT8","DBP8","FASTING_BG8","HDL8","SBP8","TC8,Albumin_urine","Hemoglobin_A1c_wholeblood","crp")
    for(i in c(1,10:14)){ 
      tmp_test[,i] <- normalization(tmp_test[,i])
    }
    model_man <- tmp_test[tmp_test$SEX == 0,colnames(tmp_test) %in% id3]
    model_man <- data.frame(model_man)
    model_man <- rownames_to_column(model_man,"ID")
    model_man$ID <-  gsub('X','',model_man$ID)
    
    model_woman <- tmp_test[tmp_test$SEX == 1,colnames(tmp_test) %in% id3]
    model_woman <- data.frame(model_woman)
    model_woman <- rownames_to_column(model_woman,"ID")
    model_woman$ID <-  gsub('X','',model_woman$ID)
    
    pdata_man <- data.frame(tmp_test$chf[tmp_test$SEX  == 0])  
    pdata_man$ID <- rownames(tmp_test[tmp_test$SEX == 0,])
    pdata_woman <- data.frame(tmp_test$chf[tmp_test$SEX  == 1]) 
    pdata_woman$ID <- rownames(tmp_test[tmp_test$SEX == 1,])
    write.table(pdata_man,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_man.txt",row.names = F)
    write.table(pdata_woman,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_JHU_woman.txt",row.names = F)
    
    write.table(FHS_man,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_FHSman_test.csv")
    write.table(FHS_woman,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_FHSwoman_test.csv")
    write.table(model_man,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_cor_lassoxgboost_man_test.csv",row.names = T,sep=",")
    write.table(model_woman,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_feature_cor_lassoxgboost_woman_test.csv",row.names = T,sep=",")
    
  }
}
#feature top5,top10,top15,top20,top25
{
  load("UMN_DMP_new.Rdata")
  load("JHU_DMP_new.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3[1:5])]
    #X5 <- subset(X5, select = c("chf",id3[1:5]))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[2] <- "target"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top5_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3[1:5])]
    #X5 <- subset(X5, select = c(id3[1:5]))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    pdata <- data.frame(JHU_DMP_new$chf)
    pdata$ID <- rownames(JHU_DMP_new)
    write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_pdata_test.txt",row.names = F)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top5_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3[1:10])]
    #X5 <- subset(X5, select = c("chf",id3[1:10]))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[3] <- "target"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top10_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3[1:10])]
    #X5 <- subset(X5, select = c(id3[1:10]))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    pdata <- data.frame(JHU_DMP_new$chf)
    pdata$ID <- rownames(JHU_DMP_new)
    #write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_pdata_test.txt",row.names = F)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top10_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3[1:15])]
    #X5 <- subset(X5, select = c("chf",id3[1:15]))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[3] <- "target"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top15_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3[1:15])]
    #X5 <- subset(X5, select = c(id3[1:15]))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    pdata <- data.frame(JHU_DMP_new$chf)
    pdata$ID <- rownames(JHU_DMP_new)
    #write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_pdata_test.txt",row.names = F)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top15_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3[1:20])]
    #X5 <- subset(X5, select = c("chf",id3[1:20]))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[3] <- "target"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top20_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3[1:20])]
    #X5 <- subset(X5, select = c(id3[1:20]))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top20_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3[1:25])]
    #X5 <- subset(X5, select = c("chf",id3[1:25]))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[3] <- "target"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top25_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3[1:25])]
    #X5 <- subset(X5, select = c(id3[1:25]))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_top25_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
  {
    X5 <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf",id3)]
    X5 <- subset(X5, select = c("chf",id3))
    X5 <- rownames_to_column(X5,"ID")
    colnames(X5)[2] <- "target"
    colnames(X5)[4] <- "AGE8"
    colnames(X5)[8] <- "Sulfonamides"
    colnames(X5)[19] <- "BMI8"
    colnames(X5)[22] <- "CREAT8"
    colnames(X5)[23] <- "Albumin_urine"
    X5 <- data.frame(X5)
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost.csv",sep="_"),row.names = F)
    
    X5 <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% c(id3)]
    X5 <- subset(X5, select = c(id3))
    X5 <- rownames_to_column(X5,"ID")
    X5 <- data.frame(X5)
    colnames(X5)[3] <- "AGE8"
    colnames(X5)[7] <- "Sulfonamides"
    colnames(X5)[18] <- "BMI8"
    colnames(X5)[21] <- "CREAT8"
    colnames(X5)[22] <- "Albumin_urine"
    write.csv(X5,paste("D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126\\20210707deepfm_feature_dmp_lassoxgboost_test.csv",sep="_"),row.names = F)
  }
}