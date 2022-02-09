library(dplyr)
setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
load(file="HFpEF.Rdata")
HFpEF <- data.frame(HFpEF)
load(file="bootstrapping_1234.Rdata")
#data_cpg,data_cpg_test
#champ prepare
{
  data_cpg = data_cpg[,c(1:7,42,55,56)]
  data_cpg$Sample_Group <- ifelse(data_cpg$chf == 1 , "chf" , "nochf")
  data_cpg_chf <- filter(data_cpg , chf == 1)
  data_cpg_nochf <- filter(data_cpg , chf == 0)
  data_cpg <- rbind(data_cpg_chf,data_cpg_nochf)
  data_cpg$Sample_Name <- paste(rep(c("chf","control"),c(nrow(data_cpg_chf),nrow(data_cpg_nochf))),c(c(1:nrow(data_cpg_chf)),c(1:nrow(data_cpg_nochf))),sep = "")
  data_cpg$Sample_Plate <- ""
  data_cpg$Pool_ID <- ""
  data_cpg_test <- filter(data_cpg,PACKS_SET == "JHU")
  data_cpg_test$SEX <- ifelse(data_cpg_test$SEX == 1,"F","M")
  write.csv(data_cpg_test,"champ_rawdata_test.csv",row.names = F)
}

#ChAMP
{
  library(ChAMP)
  library( "ggplot2" )
  library( "clusterProfiler" )
  library( "org.Hs.eg.db" )
  myLoad <- champ.load("/asnas/fangxd_group/zhaoxt/phd/cvd/methy/dbgap/c1c2_rawdata")
  beta=myLoad$beta
  save(beta,file="beta.Rdata")
  save(myLoad,file="myLoad.Rdata")
  
  champ.QC(beta=beta)
  myNorm <- champ.norm(cores=5)
  champ.QC(beta=myNorm)
  save(myNorm,file="myNorm.Rdata")
  champ.SVD(beta=myNorm)
  myCombat <- champ.runCombat(beta=myNorm,batchname="Array",pd=myLoad$pd)
  myCombat <- champ.runCombat(beta=myCombat,batchname="SEX",pd=myLoad$pd)
  champ.SVD(beta=myCombat,pd=myLoad$pd)
  save(myCombat,file="myCombat.Rdata")
  myCombat <- data.frame(myCombat)
  myRefBase <- champ.refbase(beta=myCombat,arraytype="450K")
  CorrectedBeta = myRefBase$CorrectedBeta
  CellFraction = myRefBase$CellFraction
  save(CorrectedBeta,file="CorrectedBeta.Rdata")
  save(CellFraction,file="CellFraction.Rdata")
  champ.SVD(beta=CorrectedBeta,pd=myLoad$pd)
}

{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  
  load(file= "myDMP.Rdata")#train data
  myDMP$chf_to_nochf[1:10,]
  DEG_filter <- myDMP[[1]]
  logFC_cutoff <- with( DEG_filter, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
  logFC_cutoff#0.03482952
  sigCpGs <- DEG_filter[DEG_filter$adj.P.Val<0.05 & abs(DEG_filter$logFC) > logFC_cutoff,]#318
  load(file= "test/CorrectedBeta_test.Rdata")
  new_beta_test = CorrectedBeta_test[rownames(CorrectedBeta_test) %in% rownames(sigCpGs),]
  save(new_beta_test,file="test/new_beta_test.Rdata")
  
}