#for age
#paper1==============
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load(file="UMN_meta_beta.Rdata")
UMN_cpg <- UMN_meta_beta[,-c(2:37,39:133)]
paper1 <- read.table("for age/mmc2.csv",sep=",",header = T)
{
  load(file="E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN/alldata/methylation.Rdata")
  tmp = methylation[methylation$probe.id %in% paper1$Marker, ]
  tmp = tmp[,c(1,4,10:12,21,23,25)]
  tmp = as.data.frame(tmp)
  tmp$probe.id = as.character(tmp$probe.id)
  setdiff(paper1$Marker,tmp$probe.id)#"cg25428494","ch.13.39564907R"
}
beta = UMN_cpg[,colnames(UMN_cpg) %in% c("shareid","chf",as.character(paper1$Marker))]#64 CpGs
setdiff(paper1$Marker,colnames(beta))
#"ch.2.30415474F","cg19935065","cg13001142","cg04875128","cg03473532","ch.13.39564907R"
X <- data.frame(beta)
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper1$Marker))]#64 CpGs
X <- data.frame(beta)
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1_test.csv",row.names = F,sep=",")
#paper2===================
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load(file="UMN_meta_beta.Rdata")
paper2 <- c("cg05294455","cg08598221","cg09462576","cg15804973","cg20654468","cg25268718","cg26581729","cg02867102")
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper2))]#
X <- data.frame(beta)
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age2.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper2))]#64 CpGs
X <- data.frame(beta)
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age2_test.csv",row.names = F,sep=",")
#paper3===================
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load(file="UMN_meta_beta.Rdata")
paper3 <- c("cg02228185","cg25809905","cg09809672","cg15379633","cg17861230")
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper3))]#
X <- data.frame(beta)
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age3.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper3))]#64 CpGs
X <- data.frame(beta)
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age3_test.csv",row.names = F,sep=",")
#paper4====================
#all cpg
paper_all <- c(as.character(paper1$Marker),paper2,paper3)
#duplicated(paper_all);which(duplicated(paper_all)) 
paper_all = paper_all[!duplicated(paper_all)] 
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper_all))]#
X <- data.frame(beta)
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age4.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper_all))]#64 CpGs
X <- data.frame(beta)
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age4_test.csv",row.names = F,sep=",")

#paper1 + 5 EHR#########
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper1$Marker))]#
Patient_impute <- read.csv("Patient_impute_26_nor.csv",head=T,row.names = 1)
Patient_impute <- Patient_impute[,c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")]
X <- data.frame(cbind(Patient_impute,beta))
X <- X[,c(6:7,1:5,8:72)]
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age15EHR.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper1$Marker))]#64 CpGs
Patient_impute <- read.csv("Patient_impute_test_26_nor.csv",head=T,row.names = 1)
Patient_impute <- Patient_impute[,c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")]
X <- data.frame(cbind(Patient_impute,beta))
X <- X[,c(6,1:5,7:71)]
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age15EHR_test.csv",row.names = F,sep=",")

#paper1-abs(Coefficient) > 10==============
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load(file="UMN_meta_beta.Rdata")
paper1 <- read.table("for age/mmc2.csv",sep=",",header = T)
paper1 <- filter(paper1,abs(Coefficient) >10)
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper1$Marker))]#64 CpGs
setdiff(paper1$Marker,colnames(beta))
#"cg19935065"      "ch.13.39564907R"
X <- data.frame(beta)
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1Coefficient.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper1$Marker))]#64 CpGs
X <- data.frame(beta)
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1Coefficient_test.csv",row.names = F,sep=",")

#paper1-abs(Coefficient) > 10 +5 EHR==============
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load(file="UMN_meta_beta.Rdata")
paper1 <- read.table("for age/mmc2.csv",sep=",",header = T)
paper1 <- filter(paper1,abs(Coefficient) >10)
beta = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("shareid","chf",as.character(paper1$Marker))]#64 CpGs
setdiff(paper1$Marker,colnames(beta))
#"cg19935065"      "ch.13.39564907R"
Patient_impute <- read.csv("Patient_impute_26_nor.csv",head=T,row.names = 1)
Patient_impute <- Patient_impute[,c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")]
X <- data.frame(cbind(Patient_impute,beta))
X <- X[,c(6:7,1:5,8:33)]
colnames(X)[1:2] <- c("ID","target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1Coefficient5EHR.csv",row.names = F,sep=",")
#test
load(file="test/JHU_meta_beta.Rdata")
beta = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("shareid",as.character(paper1$Marker))]#64 CpGs
Patient_impute <- read.csv("Patient_impute_test_26_nor.csv",head=T,row.names = 1)
Patient_impute <- Patient_impute[,c("AGE8","Sulfonamides","BMI8","Albumin_urine","CREAT8")]
X <- data.frame(cbind(Patient_impute,beta))
X <- X[,c(6,1:5,7:32)]
colnames(X)[1] = "ID"
pdata <- data.frame(JHU_meta_beta$chf)
pdata$ID <- rownames(JHU_meta_beta)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age1Coefficient5EHR_test.csv",row.names = F,sep=",")

#age============
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
load("UMN_DMP_new.Rdata")
X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% c("chf","AGE8")]
X <- data.frame(X)
X <- rownames_to_column(X,"ID")
#X$ID <-  gsub('X','',X$ID)
colnames(X)[2] <- c("target")#chf
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age.csv",row.names = F,sep=",")
#test
load(file="JHU_DMP_new.Rdata")
X <- JHU_DMP_new[,colnames(JHU_DMP_new) %in% "AGE8"]
X <- data.frame(X)
rownames(X) <- rownames(JHU_DMP_new)
X <- rownames_to_column(X,"ID")
X$ID <-  gsub('X','',X$ID)
colnames(X)[2] <- c("AGE8")#chf
pdata <- data.frame(JHU_DMP_new$chf)
pdata$ID <- rownames(JHU_DMP_new)
#write.table(pdata,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\20201126deepfm_pdata_test.txt",row.names = F)
write.table(X,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\data\\new_1126/20210818deepfm_feature_age_test.csv",row.names = F,sep=",")


#circlize===========
{
  library(circlize)
  library(dplyr)
  library(corrr)
  library(tibble)
  
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  id3 <- as.character(id3$Feature)
  beta1 = UMN_meta_beta[,colnames(UMN_meta_beta) %in% id3]
  beta2 = JHU_meta_beta[,colnames(JHU_meta_beta) %in% id3]
  beta = rbind(beta1,beta2)
  a = beta[,-c(1,3,4,5)] %>%
    correlate(method = "pearson") %>% 
    stretch()
  colnames(a) = c("name","feature","Freq")
  #a = a[-c(1,14:15,27:29,40:43,53:57,66:71,79:85,92:99,105:113,118:127,131:141,144:155,157:169),]
  #a = a[c(6:30,36:60,66:90,96:120,126:150),]
  a = a[a$name == "AGE8",]
  chordDiagram(as.data.frame(a[-1,]), transparency = 0.5)
  write.table(a,"for age/age_25cpg.csv",sep=",")
}
#paper1 cpg and age circlize######
paper1 <- read.table("for age/mmc2.csv",sep=",",header = T)
#paper1 <- filter(paper1,abs(Coefficient) >10)
beta1 = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("AGE8",as.character(paper1$Marker))]
beta2 = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("AGE8",as.character(paper1$Marker))]
beta = rbind(beta1,beta2)
corrr = beta %>%
  correlate(method = "pearson") %>% 
  stretch()
colnames(corrr) = c("name","feature","Freq")
corrr = corrr[corrr$name == "AGE8",]
#paper2
beta1 = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("AGE8",as.character(paper2))]
beta2 = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("AGE8",as.character(paper2))]
beta = rbind(beta1,beta2)
corrr2 = beta %>%
  correlate(method = "pearson") %>% 
  stretch()
colnames(corrr2) = c("name","feature","Freq")
corrr2 = corrr2[corrr2$name == "AGE8",]
#paper3
beta1 = UMN_meta_beta[,colnames(UMN_meta_beta) %in% c("AGE8",as.character(paper3))]
beta2 = JHU_meta_beta[,colnames(JHU_meta_beta) %in% c("AGE8",as.character(paper3))]
beta = rbind(beta1,beta2)
corrr3 = beta %>%
  correlate(method = "pearson") %>% 
  stretch()
colnames(corrr3) = c("name","feature","Freq")
corrr3 = corrr3[corrr3$name == "AGE8",]
intersect(id3,paper_all)#0
summary(a$Freq);summary(corrr$Freq);summary(corrr2$Freq);summary(corrr3$Freq)
