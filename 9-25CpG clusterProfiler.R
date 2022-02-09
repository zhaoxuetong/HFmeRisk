setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN/alldata")
#methylation <- read.csv("methylation_probe_info.csv",sep=",")
#save(methylation,file="methylation.Rdata")
load(file="methylation.Rdata")

load(file="E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN/alldata/methylation.Rdata")
setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
head(id3)
id3 = as.character(id3$Feature)
venn_CpG = methylation[methylation$probe.id %in% id3[-c(2,6,17,20,21)],]
setdiff(id3[-c(2,6,17,20,21)],venn_CpG$probe.id)
#"cg06344265"
venn_CpG$probe.id = as.character(venn_CpG$probe.id)
venn_CpG$UCSC_RefGene_Name = as.character(venn_CpG$UCSC_RefGene_Name)
venn_CpG$Relation_to_UCSC_CpG_Island = as.character(venn_CpG$Relation_to_UCSC_CpG_Island)
add_cg = c("cg06344265",NA,NA,NA,NA,NA,NA,NA,NA,37,11,NA,NA,11,NA,NA,NA,NA,NA,NA,"GRIK4",NA,"TSS200",NA,"Open sea",NA,NA,NA,NA,NA,NA,NA)
venn_CpG = rbind(add_cg,venn_CpG)
write.table(venn_CpG,"venn_probe.csv",sep=",")
{
  #all dmp
  id3 = read.csv("new_champ/myDMP.txt",sep="\t")
  id3 = rownames_to_column(id3,"probe.id")#23581
  id3 <- as.character(id3$probe.id)
  venn_CpG = methylation[methylation$probe.id %in% id3,]
  setdiff(id3,venn_CpG$probe.id)
  
  id3 = read.csv("new_champ/myDMP.txt",sep="\t")
  id3 = rownames_to_column(id3,"probe.id")#23581
  logFC_cutoff2 <- with( id3, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
  id3 <- id3[id3$adj.P.Val<0.05 & abs(id3$logFC) > logFC_cutoff2,]#318
  venn_CpG = methylation[methylation$probe.id %in% id3$probe.id,]
  setdiff(id3$probe.id,venn_CpG$probe.id)
  
}
#RefGene & Island
{
  venn_CpG$UCSC_RefGene_Group = as.character(venn_CpG$UCSC_RefGene_Group)
  venn_CpG$UCSC_RefGene_Name = as.character(venn_CpG$UCSC_RefGene_Name)
  venn_CpG$UCSC_RefGene_Group = ifelse(venn_CpG$UCSC_RefGene_Group == "", "IGR",venn_CpG$UCSC_RefGene_Group)
  venn_CpG$Relation_to_UCSC_CpG_Island = ifelse(venn_CpG$Relation_to_UCSC_CpG_Island == "", "Open sea",venn_CpG$Relation_to_UCSC_CpG_Island)
  
  #UCSC_RefGene_Group
  paste0(venn_CpG$UCSC_RefGene_Group,collapse=",")
  A<-strsplit(as.character(venn_CpG$UCSC_RefGene_Group), ";")
  tmp2 <- mapply( cbind,  A )
  df0 <- ldply (tmp2, data.frame)
  prop.table(table(df0$X..i..))
  #TSS200     IGR TSS1500    Body 1stExon   5'UTR 
  #0.050   0.200   0.250   0.275   0.050   0.175 
  #Relation_to_UCSC_CpG_Island
  paste0(venn_CpG$Relation_to_UCSC_CpG_Island,collapse=",")
  A<-strsplit(as.character(venn_CpG$Relation_to_UCSC_CpG_Island), ";")
  tmp2 <- mapply( cbind,  A )
  df1 <- ldply (tmp2, data.frame)
  prop.table(table(df1$X..i..))
  #Open sea  N_Shelf   Island  S_Shore  N_Shore  S_Shelf 
  #0.44     0.04     0.20     0.08     0.20     0.04 
  #all cpg--methylation
  methylation$UCSC_RefGene_Group = as.character(methylation$UCSC_RefGene_Group)
  methylation$Relation_to_UCSC_CpG_Island = as.character(methylation$Relation_to_UCSC_CpG_Island)
  methylation$UCSC_RefGene_Group = ifelse(methylation$UCSC_RefGene_Group == "", "IGR",methylation$UCSC_RefGene_Group)
  methylation$Relation_to_UCSC_CpG_Island = ifelse(methylation$Relation_to_UCSC_CpG_Island == "", "Open sea",methylation$Relation_to_UCSC_CpG_Island)
  
  A<-strsplit(as.character(methylation$UCSC_RefGene_Group), ";")
  tmp2 <- mapply( cbind,  A )
  df00 <- ldply (tmp2, data.frame)
  head(df00)
  prop.table(table(df00$X..i..))
  #IGR       Body      3'UTR    TSS1500      5'UTR    1stExon     TSS200 
  #0.13726348 0.35680909 0.03739231 0.15484313 0.13165007 0.06802518 0.11401675
  A<-strsplit(as.character(methylation$Relation_to_UCSC_CpG_Island), ";")
  tmp2 <- mapply( cbind,  A )
  df11 <- ldply (tmp2, data.frame)
  head(df11)
  prop.table(table(df11$X..i..))
  #Open sea    S_Shore    N_Shelf    S_Shelf    N_Shore     Island 
  #0.35281111 0.10314165 0.05029941 0.04472638 0.13191608 0.31710536 
}

#The 25 CpGs associated with HFrisk model
{
  library(ELMER)
  nearGenes <- GetNearGenes(data = mae, 
                            probes = id3[c(1,3:5,7:16,18:19,22:30)], 
                            numFlankingGenes = 20)
}
#enrichKEGG
{
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(plyr)
  library(ggplot2)
  library(forcats)
  library(enrichplot)
  library(ReactomePA)
  library(DOSE)
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(clusterProfiler)
  library(topGO)
  library(Rgraphviz)
  library(RDAVIDWebService)
  library(plyr)
  library(stringr)
  library(ggplot2)
    
  #enrichKEGG
  kegg_SYMBOL_hsa <- function(genes){ 
    gene.df <- bitr(genes, fromType = "SYMBOL",
                    toType = c("SYMBOL", "ENTREZID"),
                    OrgDb = org.Hs.eg.db)
    head(gene.df) 
    diff.kk <- enrichKEGG(gene         = gene.df$ENTREZID,
                          organism     = 'hsa',
                          pvalueCutoff = 0.99,
                          qvalueCutoff = 0.99
    )
    return( setReadable(diff.kk, OrgDb = org.Hs.eg.db,keyType = 'ENTREZID'))
  }
  paste0(venn_CpG$UCSC_RefGene_Name,collapse=",")
  A<-strsplit(as.character(venn_CpG$UCSC_RefGene_Name), ";")
  tmp2 <- mapply( cbind,  A )
  df0 <- ldply (tmp2, data.frame)
  trunk_kk=kegg_SYMBOL_hsa(df0[,1])
  trunk_df=trunk_kk@result
  write.csv(trunk_df,file = 'enrichKEGG_result.csv')
  #other
  y <- mutate(trunk_df, richFactor = Count / as.numeric(sub("/\\d+", "", BgRatio)))
  parse_ratio <- function(ratio) {
    ratio <- sub("^\\s*", "", as.character(ratio))
    ratio <- sub("\\s*$", "", ratio)
    numerator <- as.numeric(sub("/\\d+$", "", ratio))
    denominator <- as.numeric(sub("^\\d+/", "", ratio))
    return(numerator/denominator)
  }
  y2 <- mutate(y, FoldEnrichment = parse_ratio(GeneRatio) / parse_ratio(BgRatio))
  y3 <- mutate(y2, geneRatio2 = parse_ratio(GeneRatio)) %>%
    arrange(desc(GeneRatio))
  tmp <- y2[1:20,]
  ggplot(tmp, showCategory = 20,aes(FoldEnrichment, fct_reorder(Description, FoldEnrichment))) + 
    geom_segment(aes(xend=0, yend = Description)) +
    geom_point(aes(color=pvalue, size = Count)) +
    scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
    scale_size_continuous(range=c(2, 10)) +
    theme_minimal() + 
    xlab("Fold Enrichment") + #Rich Factor
    ylab(NULL) + 
    ggtitle("Enriched Pathway Ontology")
  write.csv(y2,file = 'enrichKEGG_result2.csv')
  write.csv(y3,file = 'enrichKEGG_result3.csv')
}
#enrichGO
{
  bit_gene <- function(genes){
    gene.df <- bitr(genes, fromType = "SYMBOL",
                    toType = c("SYMBOL", "ENTREZID"),
                    OrgDb = org.Hs.eg.db)
    return(gene.df$ENTREZID)
  }
  bp <- enrichGO(bit_gene( as.character(df0[,1])), ont="all",OrgDb = "org.Hs.eg.db")
  bpp = bp@result
  bp3 <- setReadable(bp, OrgDb = org.Hs.eg.db)
  barplot(bp, split="ONTOLOGY")+ facet_grid(ONTOLOGY~., scale="free")
  write.table(bp@result,"enrichGO_result.csv")
  bp = as.data.frame(bp)
  enrichMap(bp)
  goplot(bp, showCategory = 10)
  bp2 <-simplify(bp, cutoff=0.7,by="p.adjust", select_fun=min)
  as.data.frame(bp2)
  write.csv(bp2,file = 'enrichGO_simplify.csv')
}
#gometh: kegg and go;gsameth
{
  library(limma)
  library(minfi)
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  library(IlluminaHumanMethylation450kmanifest)
  library(RColorBrewer)
  library(missMethyl)
  library(matrixStats)
  library(minfiData)
  
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  
  # Get all the CpG sites used in the analysis to form the background
  load(file= "new_champ/CorrectedBeta.Rdata")
  all <-rownames(CorrectedBeta)  
  # Total number of CpG sites tested
  length(all)
  #methylGSA
  {
    library(methylGSA)
    library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
    library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
    
    DEG_filter = read.csv("new_champ/myDMP.txt",sep="\t")
    DEG_filter = rownames_to_column(DEG_filter,"probe.id")#23581
    cpg.pval <- as.data.frame(DEG_filter[DEG_filter$probe.id %in% c(id3[-c(2,6,17,20,21)]),c("probe.id","P.Value")])#514
    c = paste(DEG_filter[DEG_filter$probe.id %in% c(id3[-c(2,6,17,20,21)]),"probe.id"],DEG_filter[DEG_filter$probe.id %in% c(id3[-c(2,6,17,20,21)]),"P.Value"],sep =  "=",collapse = ",")
    cpg.pval <- c(cg20051875=3.0845273397844e-09,cg24205914=9.48317106472548e-07,cg10083824=1.23286930624815e-06,cg03556243=4.95956762541123e-06,cg03233656=1.032434350337e-05,cg21429551=1.12835004011784e-05,cg16781992=1.33529811893581e-05,cg27401945=1.34687254097115e-05,cg08614290=3.97354136093555e-05,cg05363438=4.55923591304913e-05,cg08101977=5.07348656655275e-05,cg10556349=5.36252008917427e-05,cg13352914=9.88886388394135e-05,cg11853697=0.000116194512198325,cg00045910=0.000133863261077109,cg00495303=0.000141563110724823,cg00522231=0.000155960771094521,cg06344265=0.000258547818107834,cg21024264=0.000272396328465786,cg05481257=0.000326204702802846,cg23299445=0.000569220592642687,cg05845376=0.000637513022532287,cg17766026=0.000662632200376124,cg25755428=0.00083234067533089,cg07041999=0.000990852821362967)
    head(cpg.pval)
    library(org.Hs.eg.db)
    res1 = methylglm(cpg.pval = cpg.pval, #Names should be CpG IDs.
                     minsize = 200,
                     maxsize = 500, 
                     GS.type = "KEGG")#"GO", "KEGG", or "Reactome"
    #2 gene sets
    res6 = methylglm(cpg.pval = cpg.pval, 
                     GS.type = "Reactome", 
                     minsize = 100, 
                     maxsize = 500)
    #0 gene sets
    res7 = methylglm(cpg.pval = cpg.pval, 
                     GS.type = "GO", 
                     minsize = 100, 
                     maxsize = 500)
    #449 gene sets
    res2 = methylRRA(cpg.pval = cpg.pval, 
                     method = "ORA", 
                     minsize = 200, 
                     maxsize = 500)
    #259 gene sets
    res3 = methylRRA(cpg.pval = cpg.pval, 
                     method = "GSEA", 
                     minsize = 200, 
                     maxsize = 500)
    #259 gene sets
    res4 = methylgometh(cpg.pval = cpg.pval, 
                        sig.cut = 0.05, 
                        minsize = 200, 
                        maxsize = 500)
    data(CpG2Genetoy)
    head(CpG2Gene)  
    FullAnnot = prepareAnnot(CpG2Gene) 
    data(GSlisttoy)
    GS.list = GS.list[1:10]
    res5 = methylRRA(cpg.pval = cpg.pval, 
                     FullAnnot = FullAnnot, 
                     method = "ORA", 
                     GS.list = GS.list, 
                     GS.idtype = "SYMBOL", 
                     minsize = 100, 
                     maxsize = 300)
    #10 gene sets
  }
  #par(mfrow=c(1,1))
  gst <- gometh(sig.cpg = id3[-c(2,6,17,20,21)], all.cpg=all, collection = "GO",plot.bias=TRUE)
  GO_result = topGO(gst)
  topGSA(gst, n=10)
  write.table(GO_result,"gometh_toGO.csv",sep=",")
  kegg <- gometh(sig.cpg = id3[-c(2,6,17,20,21)], all.cpg = all, collection = "KEGG", prior.prob=TRUE)
  #kegg <- gometh(sig.cpg = set, all.cpg = all, collection = "KEGG", prior.prob=TRUE)
  KEGG_result = topKEGG(kegg)
  topGSA(kegg, n=10)
  #gst.kegg.prom <- gometh(sig.cpg=id3[-c(2,6,17,20,21)], all.cpg=all, collection="KEGG",genomic.features = c("TSS200","TSS1500","1stExon"))
  #topGSA(gst.kegg.prom, n=10)
  #gst.kegg.body <- gometh(sig.cpg=id3[-c(2,6,17,20,21)], all.cpg=all, collection="KEGG", genomic.features = c("Body"))
  #topGSA(gst.kegg.body, n=10)
  write.table(KEGG_result,"gometh_toKEGG.csv",sep=",")
  
  hallmark <- readRDS(url("http://bioinf.wehi.edu.au/MSigDB/v7.1/Hs.h.all.v7.1.entrez.rds"))
  gsa <- gsameth(sig.cpg=id3[-c(2,6,17,20,21)], all.cpg=all, collection=hallmark)
  GSA_result = topGSA(gsa)
  write.table(GSA_result,"gsameth_topGSA.csv",sep=",")
  
}
#3.enrichPathway
{
  x <- enrichPathway(gene = bit_gene(df0[,1]) ,pvalueCutoff=0.95, readable=T )
  X = as.data.frame(x)
  parse_ratio <- function(ratio) {
    ratio <- sub("^\\s*", "", as.character(ratio))
    ratio <- sub("\\s*$", "", ratio)
    numerator <- as.numeric(sub("/\\d+$", "", ratio))
    denominator <- as.numeric(sub("^\\d+/", "", ratio))
    return(numerator/denominator)
  }
  
  y <- mutate(X, richFactor = Count / as.numeric(sub("/\\d+", "", BgRatio)))
  y3 <- mutate(y, geneRatio1 = parse_ratio(GeneRatio))
  y4 <- mutate(y3, geneRatio2 = parse_ratio(GeneRatio) / parse_ratio(BgRatio))
  library(ggplot2)
  library(forcats)
  library(enrichplot)
  tmp <- y4[1:20,]
  ggplot(tmp, showCategory = 20,aes(geneRatio2, fct_reorder(Description, geneRatio2))) + 
    geom_segment(aes(xend=0, yend = Description)) +
    geom_point(aes(color=pvalue, size = Count)) +
    scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
    scale_size_continuous(range=c(2, 10)) +
    theme_minimal() + 
    xlab("Fold Enrichment") +
    ylab(NULL) + 
    ggtitle("Enriched Pathway Ontology")
  tmp$log_pvalue <- -log10(tmp$pvalue)
  ggplot(tmp, showCategory = 20,aes(log_pvalue, fct_reorder(Description, log_pvalue))) + 
    geom_segment(aes(xend=0, yend = Description)) +
    geom_point(aes(size = Count)) +
    scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
    scale_size_continuous(range=c(2, 10)) +
    theme_minimal() + 
    xlab("Fold Enrichment") +
    ylab(NULL) + 
    ggtitle("Enriched Pathway Ontology")
  dotplot(x , color = "pvalue",showCategory = 20, font.size = 12, title = "")
  barplot(x, showCategory=10)
  emapplot(x,showCategory=20,color = "pvalue")
  heatplot(x,showCategory=20)
  #heatplot(x, foldChange=geneList)
  cnetplot(x, categorySize="pvalue")#foldChange=geneList
  upsetplot(x)
  write.csv(y4,file = 'enrichPathway.csv')
}
#4.enrichDO
{
  x <- enrichDO(gene          = bit_gene( df0[,1]), 
                ont           = "DO",
                pvalueCutoff  = 0.5,
                pAdjustMethod = "BH",
                universe      = x@universe,
                minGSSize     = 5,
                maxGSSize     = 500,
                qvalueCutoff  = 0.5,
                readable      = FALSE)
  x@result
  write.csv(x@result,file = 'enrichDO.csv')
  
  #plot
  emapplot(x)
  x <- setReadable(x, 'org.Hs.eg.db')
  head(x)
  cnetplot(x)
  write.csv(x,file = 'enrichDO2.csv')
}
#5.enrichNCG
{
  ncg <- enrichNCG(bit_gene(df0[,1]),pvalueCutoff = 0.95)
  head(ncg,10)
  dotplot(ncg, showCategory=30) + ggtitle("dotplot for GSEA")
  ridgeplot(ncg)
  gseaplot2(ncg, geneSetID = 1, title = ncg$Description[1])
  gseaplot2(ncg, geneSetID = 1:3)
  gseaplot2(ncg, geneSetID = 1:3, pvalue_table = TRUE,color = c("#E495A5", "#86B875", "#7DB0DD"), ES_geom = "dot")
  write.csv(ncg,file = 'enrichNCG.csv')
}
#6. enrichDGN
{
  dgn <- enrichDGN(bit_gene(df0[,1]),pvalueCutoff = 0.55)
  head(dgn,20)
  dgn <- data.frame(dgn)
  y <- mutate(dgn, richFactor = Count / as.numeric(sub("/\\d+", "", BgRatio)))
  library(ggplot2)
  library(forcats)
  library(enrichplot)
  tmp <- y[1:20,]
  ggplot(tmp, showCategory = 20,aes(richFactor, fct_reorder(Description, richFactor))) + 
    geom_segment(aes(xend=0, yend = Description)) +
    geom_point(aes(color=pvalue, size = Count)) +
    scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
    scale_size_continuous(range=c(2, 10)) +
    theme_minimal() + 
    xlab("rich factor") +
    ylab(NULL) + 
    ggtitle("Enriched Pathway Ontology")
  write.csv(y,file = 'enrichDGN.csv')
  edox <- setReadable(dgn, 'org.Hs.eg.db', 'ENTREZID')
  cnetplot(edox)
  emapplot(dgn)
}
#7.enricher
{
  library(magrittr)
  library(clusterProfiler)
  
  #data(geneList, package="DOSE")
  #gene <- names(geneList)[abs(geneList) > 2]
  wpgmtfile <- system.file("extdata/wikipathways-20180810-gmt-Homo_sapiens.gmt", package="clusterProfiler")
  #wpgmtfile <- system.file("extdata/c5.cc.v5.0.entrez.gmt", package="clusterProfiler")
  wp2gene <- read.gmt(wpgmtfile)
  head(wp2gene)
  wp2gene <- wp2gene %>% tidyr::separate(ont, c("name","version","wpid","org"), "%")
  wpid2gene <- wp2gene %>% dplyr::select(wpid, gene) #TERM2GENE
  wpid2name <- wp2gene %>% dplyr::select(wpid, name) #TERM2NAME
  ewp <- enricher(bit_gene(df0[,1]), TERM2GENE = wpid2gene, TERM2NAME = wpid2name)
  head(ewp)
  write.csv(ewp,file = 'enricher_gmt_Homo_sapiens.csv')
  #======
  gmtfile <- system.file("extdata", "c5.cc.v5.0.entrez.gmt", package="clusterProfiler")
  c5 <- read.gmt(gmtfile)
  egmt <- enricher(bit_gene(df0[,1]), TERM2GENE=c5)
  head(egmt)
}
#8.msigdbr:enricher
{
  library(msigdbr)
  
  m_df <- msigdbr(species = "Homo sapiens")
  head(m_df, 2) %>% as.data.frame
  m_t2g <- msigdbr(species = "Homo sapiens", category = "C3") %>% 
    dplyr::select(gs_name, entrez_gene)
  head(m_t2g)
  em <- enricher(bit_gene(df0[,1]), TERM2GENE=m_t2g)
  head(em)
}
#9.groupGO
{
  ggo <- groupGO(gene     = bit_gene(df0[,1]),
                 OrgDb    = org.Hs.eg.db,
                 ont      = "BP",
                 level    = 3,
                 readable = TRUE)
  
  ggo2 <- groupGO(gene     = bit_gene(df0[,1]),
                  OrgDb    = org.Hs.eg.db,
                  ont      = "CC",
                  level    = 3,
                  readable = TRUE)
  ggo3 <- groupGO(gene     = bit_gene(df0[,1]),
                  OrgDb    = org.Hs.eg.db,
                  ont      = "MF",
                  level    = 3,
                  readable = TRUE)
  
  head(ggo)
  write.csv(ggo,file = 'groupGO_BP.csv')
  write.csv(ggo2,file = 'groupGO_CC.csv')
  write.csv(ggo3,file = 'groupGO_MF.csv')
}
#10.search:KEGG
{
  library(clusterProfiler)
  search_kegg_organism('Collagen', by='kegg_code')
  ecoli <- search_kegg_organism('ece', by='scientific_name')
  dim(ecoli)
}
#11.enrichMKEGG
{
  mkk <- enrichMKEGG(gene = bit_gene(df0[,1]),organism = 'hsa')
}
#12.enrichMeSH
{
  library(meshes)
  x <- enrichMeSH(bit_gene(df0[,1]), MeSHDb = "MeSH.Hsa.eg.db", database='gendoo', category = 'C')
  head(x)
}
