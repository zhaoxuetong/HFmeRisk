#Installing 
# devtools::install_github(repo = "tiagochst/ELMER.data")
# devtools::install_github(repo = "tiagochst/ELMER")
if (!requireNamespace("BiocManager", quietly=TRUE))
  install.packages("BiocManager")
BiocManager::install("ELMER")
# BiocManager::install(version='devel')
# BiocManager::install("TCGAbiolinks")
# BiocManager::install("TCGAbiolinks", version = "2.5.9") 
# source("https://bioconductor.org/biocLite.R")
# biocLite("TCGAbiolinks")
library(ELMER)
library(MultiAssayExperiment)
library(ELMER.data)
library(sesameData)
library(tibble)
library(TCGAbiolinks)
#example
{
  distal.probes <- get.feature.probe(genome = "hg19",met.platform = "450K")
  head(distal.probes)
  data(LUSC_RNA_refined,package = "ELMER.data")
  data(LUSC_meth_refined,package = "ELMER.data")
  GeneExp[1:5,1:5]
  Meth[1:5,1:5]
  mae <- createMAE(exp = GeneExp, 
                   met = Meth,
                   save = TRUE,
                   linearize.exp = TRUE,
                   save.filename = "tmp.rda",
                   filter.probes = distal.probes,
                   met.platform = "450K",
                   genome = "hg19",
                   TCGA = TRUE)
  tmp = as.data.frame(colData(mae)[1:5,])
  View(tmp)
}
#data: RNA and methy
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/RNA/")
  load(file="./Gene.Rdata")
  load(file="./exon.Rdata")
  exon[1:4,1:4];dim(exon)
  Gene[1:4,1:4];dim(Gene)
  exon$id <- paste("sample",exon$X,sep="_")
  exon <- column_to_rownames(exon,"id")
  exon <- exon[,-c(1:2)]
  Gene$id <- paste("sample",Gene$X,sep="_")
  Gene <- column_to_rownames(Gene,"id")
  Gene <- Gene[,-c(1:2)]
  
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/new_champ/")
  load("CorrectedBeta.Rdata")
  CorrectedBeta[1:4,1:4]
  dim(CorrectedBeta)
  data_cpg <- read.csv("../champ_rawdata_train.csv")
  colnames(CorrectedBeta) <- paste("sample",data_cpg$shareid,sep="_")
  
  id = setdiff(rownames(exon),colnames(CorrectedBeta))
  exon_data = exon[!(rownames(exon) %in% id),]
  exon_data <- t(exon_data)
  exon_data[1:4,1:4];dim(exon_data)
  
  Gene_data = Gene[!(rownames(Gene) %in% id),]
  Gene_data <- t(Gene_data)
  Gene_data[1:4,1:4];dim(Gene_data)
  
  CorrectedBeta <- subset(CorrectedBeta,select = colnames(exon_data))
  dim(CorrectedBeta);dim(exon_data);dim(Gene_data)
  CorrectedBeta[1:4,1:4]
  exon_data[1:4,1:4]
  Gene_data[1:4,1:4]
  save(CorrectedBeta,exon_data,Gene_data,file="result/MAE.Rdata")
  {
    library(stringr) 
    exon_data[1:4,1:5]
    ids=data.frame(ensembl_id=str_split(rownames(exon_data),'[.]',simplify = T)[,1],
                   median=apply(exon_data,1,median))
    head(ids)
    head(ids$ensembl_id)
    library(org.Hs.eg.db)
    g2s=unique(toTable(org.Hs.egSYMBOL))
    head(g2s)
    g2e=unique(toTable(org.Hs.egENSEMBL)) 
    head(g2e)
    s2e=merge(g2e,g2s,by='gene_id')
    head(s2e)
    table(ids$ensembl_id %in% s2e$symbol)
    # FALSE  TRUE 
    # 2866 15448 
    ids=ids[ids$ensembl_id %in% s2e$symbol,]
    ids$ENSEMBL=s2e[match(ids$ensembl_id,s2e$symbol),2]
    length(unique(ids$ENSEMBL))
    head(ids) 
    ids=ids[order(ids$ENSEMBL,ids$median,decreasing = T),]
    ids=ids[!duplicated(ids$ENSEMBL),]
    dim(ids) 
    exon_data = as.data.frame(exon_data)
    exon_data = rownames_to_column(exon_data,"ensembl_id")
    exon_data = merge(ids,exon_data,by="ensembl_id")
    exon_data = column_to_rownames(exon_data,"ENSEMBL")
    exon_data = exon_data[,-c(1,2)]
    exon_data[1:4,1:4]  
    dim(exon_data)
  }
  {
    library(stringr) 
    Gene_data[1:4,1:5]
    ids=data.frame(ensembl_id=str_split(rownames(Gene_data),'[.]',simplify = T)[,1],
                   median=apply(Gene_data,1,median))
    head(ids)
    head(ids$ensembl_id)
    library(org.Hs.eg.db)
    g2s=unique(toTable(org.Hs.egSYMBOL))
    head(g2s)
    g2e=unique(toTable(org.Hs.egENSEMBL)) 
    head(g2e)
    s2e=merge(g2e,g2s,by='gene_id')
    head(s2e)
    table(ids$ensembl_id %in% s2e$symbol)
    # FALSE  TRUE 
    # 2529 14870 
    ids=ids[ids$ensembl_id %in% s2e$symbol,]
    ids$ENSEMBL=s2e[match(ids$ensembl_id,s2e$symbol),2]
    length(unique(ids$ENSEMBL))
    head(ids) 
    ids=ids[order(ids$ENSEMBL,ids$median,decreasing = T),]
    ids=ids[!duplicated(ids$ENSEMBL),]
    dim(ids) 
    Gene_data = as.data.frame(Gene_data)
    Gene_data = rownames_to_column(Gene_data,"ensembl_id")
    Gene_data = merge(ids,Gene_data,by="ensembl_id")
    Gene_data = column_to_rownames(Gene_data,"ENSEMBL")
    Gene_data = Gene_data[,-c(1,2)]
    Gene_data[1:4,1:4]  
    dim(Gene_data)
  }
  dim(Gene_data);dim(exon_data)
  save(exon_data,Gene_data,file="result/MAE_addENSEMBL.Rdata")
}
#creat mea
{
  load("result/MAE_addENSEMBL.Rdata") #exon_data,Gene_data
  distal.probes <- get.feature.probe(genome = "hg19",met.platform = "450K")
  assay <- c(rep("DNA methylation", ncol(CorrectedBeta)),
             rep("Gene expression", ncol(exon_data)))
  primary <- c(colnames(CorrectedBeta),colnames(exon_data))
  data_cpg <- read.csv("../champ_rawdata_train.csv")
  data_cpg$shareid <- paste("sample",data_cpg$shareid,sep="_")
  colname <- data_cpg[colnames(CorrectedBeta) %in% data_cpg$shareid ,"Sample_Group"]
  colname <- c(colnames(CorrectedBeta),colnames(exon_data))
  sampleMap <- data.frame(assay,primary,colname)
  #distal.probes <- get.feature.probe(genome = "hg19",met.platform = "450K")
  colData <- data.frame(sample = colnames(CorrectedBeta))
  tmp = data_cpg[data_cpg$shareid %in% colData$sample ,]
  colData <- merge(colData,tmp[,c(1,11)],by.x="sample",by.y="shareid")
  colData = unique(colData)
  rownames(colData) <- colData$sample
  mae <- createMAE(exp = exon_data, 
                   met = CorrectedBeta,
                   save = TRUE,
                   filter.probes = distal.probes,
                   colData = colData,
                   #sampleMap = sampleMap,
                   linearize.exp = TRUE,
                   save.filename = "mae.rda",
                   met.platform = "450K",
                   genome = "hg19",
                   TCGA = FALSE)
  save(mae,"result/mae.rda")
}
#DMP
{
  mae <- get(load("result/mae.rda"))
  sig.diff <- get.diff.meth(data = mae, 
                            group.col = "Sample_Group",
                            group1 =  "chf",
                            group2 = "nochf",
                            # if supervised mode set to 1
                            mode = "supervised",
                            minSubgroupFrac = 1, 
                            sig.dif = 0,
                            diff.dir = "both", # “hypo” and “hyper” 
                            cores = 1, 
                            dir.out ="result", 
                            save =  FALSE ,
                            pvalue = 0.05#0.05#1
                            )
  head(sig.diff);
  dim(sig.diff)#pvalue = 0.05,19;0.5,8537
  write.table(sig.diff,"result/getMethdiff.both.probes.csv")#pvalue = 0.5,8537
  write.table(sig.diff,"result/getMethdiff.both.probes.significant.csv")#pvalue = 0.05,19
  ##"getMethdiff.hypo.probes.csv"            
  ##"getMethdiff.hypo.probes.significant.csv"
  save(sig.diff,file="result/sig_diff.Rdata")#pvalue = 0.5,8537
}
#Identifying putative probe-gene pairs
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/new_champ/")
  mae <- get(load("result/mae.rda"))
  #sig.diff <- read.csv("result/getMethdiff.both.probes.significant.csv",sep = " ")
  nearGenes <- GetNearGenes(data = mae, 
                            probes = sig.diff$probe, 
                            numFlankingGenes = 20) # 10 upstream and 10 dowstream genes
  save(nearGenes,file="result/nearGenes.Rdata")
  Hypo.pair <- get.pair(data = mae,
                        group.col = "Sample_Group",
                        group1 =  "chf",
                        group2 = "nochf",
                        nearGenes = nearGenes,
                        #minSubgroupFrac = 1,
                        mode = "unsupervised",#supervised or unsupervised
                        permu.dir = "result/permu",
                        #Please set to 100000 to get significant results
                        #permu.size = 100000,
                        raw.pvalue = 0.05,#1,
                        #Please set to 0.001 to get significant results
                        Pe = 0.05,#1, 
                        cores = 1,
                        #See preAssociationProbeFiltering function
                        filter.probes = FALSE, 
                        # filter.percentage = 0.01,
                        # filter.portion = 0.3,
                        dir.out = "result",
                        label = "both",
                        save = FALSE)
  dim(Hypo.pair)
  write.table(Hypo.pair,"result/getPair.both.all.pairs.statistic5.csv")
  write.table(Hypo.pair,"result/getPair.both.all.pairs.statistic.significant.csv")
  #"getPair.hypo.all.pairs.statistic.csv"                  
  #"getPair.hypo.pairs.significant.csv"                    
  #"getPair.hypo.pairs.statistic.with.empirical.pvalue.csv"
}
#motif
{
  # Load results from previous sections
  mae <- get(load("result/mae.rda"))
  Hypo.pair1 = read.csv("result/getPair.both.all.pairs.statistic1.csv",sep=" ")
  Hypo.pair2 = read.csv("result/getPair.both.all.pairs.statistic2.csv",sep=" ")
  Hypo.pair3 = read.csv("result/getPair.both.all.pairs.statistic3.csv",sep=" ")
  Hypo.pair4 = read.csv("result/getPair.both.all.pairs.statistic4.csv",sep=" ")
  Hypo.pair5 = read.csv("result/getPair.both.all.pairs.statistic5.csv",sep=" ")
  pair <- rbind(rbind(rbind(rbind(Hypo.pair1,Hypo.pair2),Hypo.pair3),Hypo.pair4),Hypo.pair5)
  head(pair) # significantly hypomethylated probes with putative target genes
  pair = Hypo.pair
  # Identify enriched motif for significantly hypomethylated probes which 
  # have putative target genes.
  enriched.motif <- get.enriched.motif(data = mae,
                                       #probes = pair$Probe, 
                                       probes = sig.diff$probe, 
                                       dir.out = "result", 
                                       label = "both",
                                       min.incidence = 0,
                                       lower.OR = 1,
                                       save = FALSE)
  names(enriched.motif)
  head(enriched.motif[names(enriched.motif)[1]]) ## probes in the given set that have the first motif.
  write.table(enriched.motif$FOSL2_HUMAN.H11MO.0.A,"result/getMotif.both.motif.enrichment.txt")
  save(enriched.motif,file="result/getMotif.both.enriched.motifs.rda")
  #"getMotif.hypo.enriched.motifs.rda"  "getMotif.both.motif.enrichment.csv" "motif.enrichment.pdf") 
}
#TF
{
  # Load results from previous sections
  mae <- get(load("mae.rda"))
  load("result/getMotif.both.enriched.motifs12.rda")
  TF <- get.TFs(data = mae, 
                group.col = "Sample_Group",
                group1 =  "chf",
                group2 = "nochf",
                mode = "unsupervised",
                enriched.motif = enriched.motif,
                dir.out = "result", 
                cores = 1, 
                label = "both",
                save =  FALSE )
  ##"getTF.hypo.TFs.with.motif.pvalue.rda"             
  ##"getTF.hypo.significant.TFs.with.motif.summary.csv"
  write.table(TF,"result/getTF.both.significant.TFs.with.motif.summary.csv")
  save(TF,file="result/getTF.both.TFs.with.motif.pvalue.rda")
}
#Scatter plots
{
  mae <- get(load("mae.rda"))
  load("result/getMotif.hypo.enriched.motifs.rda")
  scatter.plot(data = mae,
               byProbe = list(probe = c("cg27401945"), numFlankingGenes = 20), 
               category = "Sample_Group", 
               lm = TRUE, # Draw linear regression curve
               save = TRUE) 
  scatter.plot(data = mae,
               byPair = list(probe = c("cg27401945"), gene = c("ENSG00000148704")), 
               category = "Sample_Group", save = TRUE, lm_line = TRUE) 
  
  load("result/getMotif.hypo.enriched.motifs.rda")
  names(enriched.motif)[1]
  scatter.plot(data = mae,
               byTF = list(TF = c("VAX1","DIP2C"),
                           probe = enriched.motif[[names(enriched.motif)[2]]]), 
               category = "Sample_Group",
               save = TRUE, 
               lm_line = TRUE)
}
#XX 
{
  # Load results from previous sections
  mae <- get(load("mae.rda"))
  #pair <- read.csv("result/getPair.hypo.pairs.significant.csv")
  schematic.plot(pair = pair, 
                 data = mae,
                 group.col = "Sample_Group",
                 byProbe = pair$Probe[1],
                 save = FALSE)
  schematic.plot(pair = pair, 
                 data = mae,   
                 group.col = "Sample_Group", 
                 byGene = pair$GeneID[1],
                 save = FALSE)
}
#XX motif
{
  motif.enrichment.plot(motif.enrichment = enriched.motif,
                        significant = list(OR = 1.5,lowerOR = 1.3),
                        label = "both",
                        save = FALSE)
  motif.enrichment.plot(motif.enrichment = "result/getMotif.hypo.motif.enrichment.csv",
                        significant = list(OR = 1.5,lowerOR = 1.3),
                        label = "hypo",
                        summary = TRUE,
                        save = FALSE)  
}
#TF
{
  load("result/getTF.hypo.TFs.with.motif.pvalue.rda")
  motif <- colnames(TF.meth.cor)[1]
  TF.rank.plot(motif.pvalue = TF.meth.cor,
               motif = motif,
               save = FALSE)
}
#XX heatmap
{
  # Load results from previous sections
  mae <- get(load("mae.rda"))
  pair <- read.csv("result/getPair.hypo.pairs.significant.csv")
  heatmapPairs(data = mae, 
               group.col = "Sample_Group",
               group1 = "Chf", 
               #annotation.col = c("years_smoked","gender"),
               group2 = "Nochf",
               pairs = pair,
               filename =  NULL)
}

{
  library(plyr)
  A<-strsplit(as.character(names(enriched.motif)), "_")
  tmp2 <- mapply( cbind,  A )
  df0 <- ldply (tmp2[1,], data.frame)
  #median level of methylation estimated from all distal probes
  length(enriched.motif$FOSL2_HUMAN.H11MO.0.A)
  FOSL2 = enriched.motif$FOSL2_HUMAN.H11MO.0.A
  FOSL1 = enriched.motif$FOSL1_HUMAN.H11MO.0.A
}