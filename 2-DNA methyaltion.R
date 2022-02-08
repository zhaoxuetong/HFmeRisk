library(dplyr)
setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
load(file="HFpEF.Rdata")
HFpEF <- data.frame(HFpEF)
data_cpg = filter(HFpEF,PACKS_SET == "UMN")
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
  data_cpg_train <- filter(data_cpg,PACKS_SET == "UMN")
  write.csv(data_cpg_train,"champ_rawdata_train.csv",row.names = F)
}

#ChAMP
{
  library(ChAMP)
  library( "ggplot2" )
  library( "clusterProfiler" )
  library( "org.Hs.eg.db" )
  myLoad <- champ.load("/asnas/group/phd/cvd/methy/dbgap/c1c2_rawdata")
  beta=myLoad$beta
  # save(beta,file="beta.Rdata")
  # save(myLoad,file="myLoad.Rdata")
  # champ.QC(beta=beta)
  myNorm <- champ.norm(cores=5)
  # champ.QC(beta=myNorm)
  # save(myNorm,file="myNorm.Rdata")
  # champ.SVD(beta=myNorm)
  myCombat <- champ.runCombat(beta=myNorm,batchname="Array",pd=myLoad$pd[,-1])
  myCombat <- champ.runCombat(beta=myCombat,batchname="SEX",pd=myLoad$pd[,-1])
  # champ.SVD(beta=myCombat,pd=myLoad$pd)
  # save(myCombat,file="myCombat.Rdata")
  myCombat <- data.frame(myCombat)
  myRefBase <- champ.refbase(beta=myCombat,arraytype="450K")
  CorrectedBeta = myRefBase$CorrectedBeta
  CellFraction = myRefBase$CellFraction
  save(CorrectedBeta,file="CorrectedBeta.Rdata")
  save(CellFraction,file="CellFraction.Rdata")
  
  champ.SVD(beta=CorrectedBeta,pd=myLoad$pd)
  myDMP <- champ.DMP(beta=CorrectedBeta) #beta = myNorm
  #myDMP_H <- champ.DMP(beta=CorrectedBeta, pheno=myLoad$pd$Sample_Group, compare.group=c("oxBS", "BS"))
  # In above code, you can set compare.group() as "oxBS" and "BS" to do DMP detection between hydroxymethylatio and normal methylation.
  #hmc <- myDMP[[1]][myDMP[[1]]$deltaBeta>0,]
  # Then you can use above code to extract hydroxymethylation CpGs.
  dim(myDMP)
  dim(myDMP[[1]])
  write.table(myDMP[[1]],file="myDMP.txt",sep = "\t",quote = F,row.names = T)
  myDMR <- champ.DMR(beta=CorrectedBeta,method="ProbeLasso") #ProbeLasso is dmrP
  myDMR_Bumphunter <- champ.DMR(beta=CorrectedBeta,method="Bumphunter")
  myDMR_DMRcate <- champ.DMR(beta=CorrectedBeta,method="DMRcate")
  write.table(myDMR$ProbeLassoDMR,file="myDMR_ProbeLasso.txt",sep = "\t",quote = F,row.names = T)
  write.table(myDMR_Bumphunter$BumphunterDMR,file="myDMR_Bumphunter.txt",sep = "\t",quote = F,row.names = T)
  write.table(myDMR_DMRcate$DMRcateDMR,file="myDMR_DMRcate.txt",sep = "\t",quote = F,row.names = T)
}
 
