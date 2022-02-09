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
  myLoad <- champ.load("/asnas/fangxd_group/zhaoxt/phd/cvd/methy/dbgap/c1c2_rawdata")
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
#other DMR
{
  library(limma)
  library(minfi)
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  library(IlluminaHumanMethylation450kmanifest)
  library(RColorBrewer)
  library(missMethyl)
  library(matrixStats)
  library(minfiData)
  library(Gviz)
  library(DMRcate)
  library(stringr)
  #DMR
  individual <- factor(myLoad$pd$Sample_Group)
  design <- model.matrix(~0+individual, data=myLoad$pd)
  colnames(design) <- c(levels(individual))
  contMatrix <- makeContrasts(chf - nochf,levels=design)
  myAnnotation <- cpg.annotate(object = CorrectedBeta, datatype = "array", what = "Beta", 
                               analysis.type = "differential", design = design, 
                               contrasts = TRUE, cont.matrix = contMatrix, 
                               coef = "chf - nochf", arraytype = "450K")
  DMRs <- dmrcate(myAnnotation, lambda=1000, C=2)
  head(DMRs$results)
  # convert the regions to annotated genomic ranges
  data(dmrcatedata)
  results.ranges <- extractRanges(DMRs, genome = "hg19")
  write.table(results.ranges,"results.ranges.txt")
  # pal <- brewer.pal(8,"Dark2")
  # groups <- pal[1:length(unique(individual))]
  # names(groups) <- levels(factor(individual))
  # cols <- groups[as.character(factor(individual))]
  # samps <- 1:nrow(myLoad$pd)
  # par(mfrow=c(1,1))
  # DMR.plot(ranges=results.ranges, 
  #          dmr=1, 
  #          CpGs=CorrectedBeta, 
  #          phen.col=cols, 
  #          what = "Beta",
  #          arraytype = "450K", 
  #          pch=16, 
  #          toscale=TRUE, 
  #          plotmedians=TRUE, 
  #          #genome="hg19", 
  #          samps=samps)
  #DMV
  individual <- factor(myLoad$pd$Sample_Group)
  design <- model.matrix(~0+individual, data=myLoad$pd)
  colnames(design) <- c(levels(individual))
  fitvar <- varFit(CorrectedBeta, design = design, coef = c(1,2))
  summary(decideTests(fitvar))
  allDV <- topVar(fitvar, coef=2, number = nrow(CorrectedBeta))
  write.table(allDV,"allDV.txt")
  topDV <- topVar(fitvar, coef=2, number = 10000)
  write.table(topDV,"topDMV.txt")
  cpgsDV <- rownames(topDV)
  par(mfrow=c(2,2))
  for(i in 1:4){
    stripchart(#CorrectedBeta[rownames(CorrectedBeta)==cpgsDV[i],c(1:87,sample(88:968,87))]~design[c(1:87,sample(88:968,87)),2],
               CorrectedBeta[rownames(CorrectedBeta)==cpgsDV[i],]~design[,2],
               method="jitter",
               group.names=c("chf","nochf"),
               pch=16,
               cex=1.5,
               col=c(4,2),
               ylab="Beta values",
               vertical=TRUE,
               cex.axis=1.5,
               cex.lab=1.5)
    title(cpgsDV[i],cex.main=1.5)
  }
  
  #GO
  DMPs = read.csv("myDMP.txt",sep="\t",header = T)
  sigCpGs <- rownames(DMPs[DMPs$adj.P.Val < 0.001,])
  length(sigCpGs)
  all <- rownames(DMPs)
  length(all)
  par(mfrow=c(1,1))
  load("/asnas/fangxd_group/zhaoxt/phd/CVD_Databes/human_c2_v5.rdata")
  # load Broad human curated (C2) gene sets
  #"E:/hening_workplace/R_library/methylationArrayAnalysis/extdata"
  gsa <- gsameth(sig.cpg=sigCpGs, all.cpg=all, collection=Hs.c2)
  write.table(gsa, "gsa_result.txt")
  gst <- gometh(sig.cpg=sigCpGs, 
                all.cpg=all,  #background
                collection="GO",
                plot.bias=TRUE)
  write.table(gst,"gst_result.txt")
}

{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/")
  load(file="E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN/alldata/methylation.Rdata")
  library(tibble)
  #DMV 
  {
    allDMV = read.csv("new_champ/allDV.txt",sep=" ")
    allDMV = rownames_to_column(allDMV,"probe.id")
    Var_cutoff <- with( allDMV, mean( abs( SampleVar ) ) + 2 * sd( abs( SampleVar ) ) )
    Var_cutoff = 0.05#0.01070459
    sigDMV <- allDMV[allDMV$Adj.P.Value<0.01 & abs(allDMV$SampleVar) > Var_cutoff,]#5930
    write.table(sigDMV,"new_champ/sigDMV.txt")
  }
  
  #DMP
  {
    DEG_filter = read.csv("new_champ/myDMP.txt",sep="\t")
    DEG_filter = rownames_to_column(DEG_filter,"probe.id")#23581
    logFC_cutoff <- with( DEG_filter, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff#0.03324818
    sigCpGs <- DEG_filter[DEG_filter$adj.P.Val<0.05 & abs(DEG_filter$logFC) > logFC_cutoff,]#514
    write.table(sigCpGs,"new_champ/sigCpGs.txt")
  }
  
  data = merge(sigDMV[,c(1,2,7)],sigCpGs[,c(1,2,6,11,12,15:17)],by="probe.id")#109
  write.table(data,"new_champ/sigCpGs_sigDVP.txt")
  #Bumphunter
  {
    dmr_Bumphunter_early = read.csv("new_champ/myDMR_Bumphunter.txt",sep="\t")#327
    dmr_Bumphunter_early$chr = gsub('chr','',dmr_Bumphunter_early$seqnames)
    #dmp in Bumphunter
    Bumphunter_dmp_early1 = data.frame()#1218
    for(i in 1:nrow(dmr_Bumphunter_early)){
      tmp_ = sigCpGs_early[dmr_Bumphunter_early$chr[i] == sigCpGs_early$CHR & dmr_Bumphunter_early$start[i] < sigCpGs_early$MAPINFO & dmr_Bumphunter_early$end[i] > sigCpGs_early$MAPINFO,]
      Bumphunter_dmp_early1 = rbind(Bumphunter_dmp_early1,tmp_)
    }
    #dmp in Bumphunter
    Bumphunter_dmp_early = data.frame()#1218
    for(i in 1:nrow(dmr_Bumphunter_early)){
      tmp_ = DEG_filter_early[dmr_Bumphunter_early$chr[i] == DEG_filter_early$CHR & dmr_Bumphunter_early$start[i] < DEG_filter_early$MAPINFO & dmr_Bumphunter_early$end[i] > DEG_filter_early$MAPINFO,]
      Bumphunter_dmp_early = rbind(Bumphunter_dmp_early,tmp_)
    }
    #all dmp in DMRcate
    Bumphunter_all_early = data.frame()#13960
    for(i in 1:nrow(dmr_Bumphunter_early)){
      tmp_ = methylation[dmr_Bumphunter_early$chr[i] == methylation$CHR & dmr_Bumphunter_early$start[i] < methylation$MAPINFO & dmr_Bumphunter_early$end[i] > methylation$MAPINFO,]
      Bumphunter_all_early = rbind(Bumphunter_all_early,tmp_)
    }
  }
  dim(Bumphunter_dmp_early1);dim(Bumphunter_dmp_early);dim(Bumphunter_all_early)
  #DMRcate
  {
    dmr_DMRcate_early = read.csv("new_champ/myDMR_DMRcate.txt",sep="\t")#327
    dmr_DMRcate_early$chr = gsub('chr','',dmr_DMRcate_early$seqnames)
    #dmp in DMRcate
    DMRcate_dmp_early1 = data.frame()#1218
    for(i in 1:nrow(dmr_DMRcate_early)){
      tmp_ = sigCpGs_early[dmr_DMRcate_early$chr[i] == sigCpGs_early$CHR & dmr_DMRcate_early$start[i] < sigCpGs_early$MAPINFO & dmr_DMRcate_early$end[i] > sigCpGs_early$MAPINFO,]
      DMRcate_dmp_early1 = rbind(DMRcate_dmp_early1,tmp_)
    }
    #dmp in DMRcate
    DMRcate_dmp_early = data.frame()#1218
    for(i in 1:nrow(dmr_DMRcate_early)){
      tmp_ = DEG_filter_early[dmr_DMRcate_early$chr[i] == DEG_filter_early$CHR & dmr_DMRcate_early$start[i] < DEG_filter_early$MAPINFO & dmr_DMRcate_early$end[i] > DEG_filter_early$MAPINFO,]
      DMRcate_dmp_early = rbind(DMRcate_dmp_early,tmp_)
    }
    #all dmp in DMRcate
    DMRcate_all_early = data.frame()#13960
    for(i in 1:nrow(dmr_DMRcate_early)){
      tmp_ = methylation[dmr_DMRcate_early$chr[i] == methylation$CHR & dmr_DMRcate_early$start[i] < methylation$MAPINFO & dmr_DMRcate_early$end[i] > methylation$MAPINFO,]
      DMRcate_all_early = rbind(DMRcate_all_early,tmp_)
    }
  }
  dim(DMRcate_dmp_early1);dim(DMRcate_dmp_early);dim(DMRcate_all_early)
  #ProbeLasso
  {
    dmr_ProbeLasso_early = read.csv("new_champ/myDMR_ProbeLasso.txt",sep="\t")#327
    dmr_ProbeLasso_early$chr = gsub('chr','',dmr_ProbeLasso_early$seqnames)
    #dmp in ProbeLasso
    ProbeLasso_dmp_early1 = data.frame()#1218
    for(i in 1:nrow(dmr_ProbeLasso_early)){
      tmp_ = sigCpGs_early[dmr_ProbeLasso_early$chr[i] == sigCpGs_early$CHR & dmr_ProbeLasso_early$start[i] < sigCpGs_early$MAPINFO & dmr_ProbeLasso_early$end[i] > sigCpGs_early$MAPINFO,]
      ProbeLasso_dmp_early1 = rbind(ProbeLasso_dmp_early1,tmp_)
    }
    #dmp in ProbeLasso
    ProbeLasso_dmp_early = data.frame()#1218
    for(i in 1:nrow(dmr_ProbeLasso_early)){
      tmp_ = DEG_filter_early[dmr_ProbeLasso_early$chr[i] == DEG_filter_early$CHR & dmr_ProbeLasso_early$start[i] < DEG_filter_early$MAPINFO & dmr_ProbeLasso_early$end[i] > DEG_filter_early$MAPINFO,]
      ProbeLasso_dmp_early = rbind(ProbeLasso_dmp_early,tmp_)
    }
    #all dmp in DMRcate
    ProbeLasso_all_early = data.frame()#13960
    for(i in 1:nrow(dmr_ProbeLasso_early)){
      tmp_ = methylation[dmr_ProbeLasso_early$chr[i] == methylation$CHR & dmr_ProbeLasso_early$start[i] < methylation$MAPINFO & dmr_ProbeLasso_early$end[i] > methylation$MAPINFO,]
      ProbeLasso_all_early = rbind(ProbeLasso_all_early,tmp_)
    }
  }
  dim(ProbeLasso_dmp_early1);dim(ProbeLasso_dmp_early);dim(ProbeLasso_all_early)
}