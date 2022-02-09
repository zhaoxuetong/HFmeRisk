library("readxl")

#RNA:c1+c2
#after try only training data, result is good than now
{
  setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/RNA/")
  load(file="./Gene.Rdata")
  load(file="./exon.Rdata")
  
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  load("train_data.Rdata")
  load("test/test_data.Rdata")
  dat1 = read.csv("champ_rawdata_train.csv",header = T)
  dat2 = read.csv("champ_rawdata_test.csv",header = T,skip = 7)
  train = cbind(dat1[,c(1,12)],train_data)
  test = cbind(dat2[,c(1,12)],test_data)
  
  train_data_gene <- merge(train[,c("shareid","Heart.failure")],Gene,by.y="X",by.x="shareid")#654
  test_data_gene <- merge(test[,c("shareid","Heart.failure")],Gene,by.y="X",by.x="shareid")#164
  train_data_exon <- merge(train[,c("shareid","Heart.failure")],exon,by.y="X",by.x="shareid")#654
  test_data_exon <- merge(test[,c("shareid","Heart.failure")],exon,by.y="X",by.x="shareid")#164
  train_data_gene = train_data_gene[,-3];test_data_gene = test_data_gene[,-3]
  train_data_exon = train_data_exon[,-3];test_data_exon = test_data_exon[,-3]
  dim(train_data_gene);dim(test_data_gene)
  dim(train_data_exon);dim(test_data_exon)
  # 654 + 164 = 818
  
  all_gene <- rbind(train_data_gene,test_data_gene)
  all_gene <- train_data_gene
  all_exon <- rbind(train_data_exon,test_data_exon)
  all_exon <- train_data_exon
  dim(all_gene);dim(all_exon)
  save(all_gene,all_exon,file="new_champ/gene_exon.Rdata")
  cpg_gene <- c("GRIK4","MRI1","RHOBTB1","GARS","CYP2E1","KCNIP4","CDH4","GRM4" ,
                "VIPR2","CACNA1H" ,"SLC25A2","GDF7" , "FBXO28","ZBTB20" ,"SLC1A4",
                "ITGB1BP1","DLGAP1","PTF1A","MYOM2","DIP2C","FOXD3","HIF1AN","DYRK2",
                "ADPGK","VAX1")
  result1 <- all_gene[,colnames(all_gene) %in% c(cpg_gene,"shareid","Heart.failure")]
  result2 <- all_exon[,colnames(all_exon) %in% c(cpg_gene,"shareid","Heart.failure")]
  colnames(result1)
  result1 <- all_gene
  result2 <- all_exon
  #"FBXO28"   "GDF7"     "SLC1A4"   "ITGB1BP1" "ZBTB20"   "KCNIP4"  
  #"SLC25A2"  "GRM4"     "GARS"     "VIPR2"    "CYP2E1"   "RHOBTB1"   
  #"GRIK4"    "CACNA1H"  "DLGAP1"   "CDH4"
  #missing: "MRI1"
  result = result2 #result2
  #ggstatsplot
  {
    library( "ggstatsplot" )
    #load( './final_exprSet.Rdata' )
    special_gene = c("GRIK4","RHOBTB1","GARS","CYP2E1","KCNIP4","CDH4","GRM4" ,
                     "VIPR2","CACNA1H" ,"SLC25A2","GDF7" , "FBXO28","ZBTB20" ,"SLC1A4",
                     "ITGB1BP1","DLGAP1")
    for( gene in special_gene ){
      filename <- paste("diff_", gene, '.pdf', sep = '' )
      TMP = result[ ,colnames( result ) == "RHOBTB1"]
      data = as.data.frame(TMP)
      data$group = result$chf.x
      p <- ggbetweenstats(data = data, x = group,  y = TMP )
      p
      ggsave( p, filename = filename)
    }
  }
  #ggboxplot
  {
    result$target <- ifelse(result$Heart.failure == 1,"Heart.failure","no-Heart.failure")
    #Gene_c1$target <- ifelse(Gene_c1$chf == 1,"chf","no-chf")
    library(ggpubr)
    library(cowplot)
    my_comparisons <- list(c("Heart.failure", "no-Heart.failure") )
    # Create the box plot. Change colors by groups: Species
    # Add jitter points and change the shape by RPS6KA2groups
    plot_result <- lapply(c(colnames(result)[-c(1,2,27)]), function(x){
      plot_grid(ggboxplot(
        result, x = "target", y = x,
        color = "target", palette = c("#00AFBB", "#E7B800"),
        add = "jitter"
      )+stat_compare_means(comparisons = my_comparisons, method = "t.test"))
    })
    plot_grid(plotlist=plot_result, ncol=8)
    ggsave(file="ggboxplot_exon.pdf",width=25.5, height=25.6)
  }
  
  #gene
  {
    all_gene[1:5,1:5]
    all_gene <- data.frame(all_gene)
    all_gene <- all_gene[,-2]
    colnames(all_gene)[2] <- c("chf")
    
    library(limma)
    group_list=factor(all_gene$chf)
    n_expr = all_gene[grep( "1", group_list ),]
    g_expr = all_gene[grep( "0", group_list ),]
    exprSet = rbind( n_expr, g_expr )
    rownames(exprSet) <- exprSet$X
    exprSet = exprSet[,-c(1,2)]
    exprSet = t(exprSet)
    exprSet[1:4,1:4]
    group_list = c(rep( 'chf', nrow( n_expr ) ), 
                   rep( 'nochf',nrow( g_expr ) ) )
    
    design <- model.matrix(~0 + factor(group_list)) 
    colnames(design) = levels(factor(group_list)) 
    rownames(design) = colnames(exprSet)   ## 替换行名为样本ID
    head(design )
    
    contrast.matrix <- makeContrasts(paste0(unique(group_list),collapse = "-"), levels = design) 
    contrast.matrix  
    
    ####### 做DEG差异分析
    exprSet[1:4,1:4]
    fit <- lmFit(exprSet, design)   ## lmFit看名字就是线性拟合
    fit2 <- contrasts.fit(fit, contrast.matrix)  ## 计算标准误差
    fit2 <- eBayes( fit2 )   ## Empirical Bayes Statistics for Differential Expression
    nrDEG = topTable( fit2,adjust='fdr', coef = 1, n = Inf )
    write.table( nrDEG,"./all_gene_DEG_UMNJHU.csv",sep=",")
    head(nrDEG)
    nrDEG_ <- filter(nrDEG,P.Value < 0.05)
    summary(nrDEG_$logFC)
    logFC_cutoff <- with( nrDEG_, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff#0.1469162
  }
  #exon
  {
    all_exon[1:5,1:5]
    all_exon <- data.frame(all_exon)
    all_exon <- all_exon[,-2]
    colnames(all_exon)[2] <- c("chf")
    
    library(limma)
    group_list=factor(all_exon$chf)
    n_expr = all_exon[grep( "1", group_list ),]
    g_expr = all_exon[grep( "0", group_list ),]
    exprSet = rbind( n_expr, g_expr )
    rownames(exprSet) <- exprSet$X
    exprSet = exprSet[,-c(1,2)]
    exprSet = t(exprSet)
    exprSet[1:4,1:4]
    group_list = c(rep( 'chf', nrow( n_expr ) ), 
                   rep( 'nochf',    nrow( g_expr ) ) )
    
    design <- model.matrix(~0 + factor(group_list)) 
    colnames(design) = levels(factor(group_list)) 
    rownames(design) = colnames(exprSet)   ## 替换行名为样本ID
    design  
    
    contrast.matrix <- makeContrasts(paste0(unique(group_list),collapse = "-"), levels = design) 
    contrast.matrix  
    
    ####### 做DEG差异分析
    exprSet[1:4,1:4]
    fit <- lmFit(exprSet, design)   ## lmFit看名字就是线性拟合
    fit2 <- contrasts.fit(fit, contrast.matrix)  ## 计算标准误差
    fit2 <- eBayes( fit2 )   ## Empirical Bayes Statistics for Differential Expression
    nrDEG = topTable( fit2, coef = 1, n = Inf )
    write.table( nrDEG,"./all_exon_DEG_UMNJHU.csv",sep=",")
    head(nrDEG)
    nrDEG_ <- filter(nrDEG,P.Value < 0.05)
    summary(nrDEG_$logFC)
    logFC_cutoff <- with( nrDEG_, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff #0.202579
    
  }
  {
    library(org.Hs.eg.db)
    library(clusterProfiler)
    library(enrichplot)
    exon_DEG <- read.csv("all_exon_DEG_UMNJHU.csv")
    gene_DEG <- read.csv("all_gene_DEG_UMNJHU.csv")
    #data = exon_DEG
    data = gene_DEG
    logFC_cutoff <- with( data, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff #0.0961043,0.07283755
    data <- data[abs(data$logFC) > logFC_cutoff,]
    data <- data[data$P.Value < 0.05,] #491，449
    df <- bitr( rownames(data), fromType = "SYMBOL", toType = c( "ENTREZID" ,"ENSEMBL"), OrgDb = org.Hs.eg.db )
    head( df )
    data <- rownames_to_column(data,"Gene")
    df <- merge(df,data,by.x="SYMBOL",by.y="Gene")
    head( df )
    #df$logFC <- data$logFC[match(df$SYMBOL,data$Gene)]
    geneList=df$logFC
    names(geneList)=df$ENTREZID
    geneList=sort(geneList,decreasing = T)
    head(geneList)
    
    #gseKEGG
    {
      kk_gse <- gseKEGG(geneList     = geneList,
                        organism     = 'hsa',
                        nPerm        = 1000,
                        minGSSize    = 10,
                        pvalueCutoff = 0.95,
                        verbose      = FALSE)
      kk_gse@result
      write.table(kk_gse@result,"gesKEGG_gene_UMNJHU.csv")
      write.table(kk_gse@result,"gesKEGG_exon_UMNJHU.csv")
      #plot
      #exon
      paper_choose <- c('NOD-like receptor signaling pathway',
                        'Transcriptional misregulation in cancer',
                        "Viral protein interaction with cytokine and cytokine receptor",
                        "Herpes simplex virus 1 infection")
      #gene
      paper_choose <- c('Cytokine-cytokine receptor interaction',
                        'Hepatitis C',
                        "Metabolic pathways",
                        "Influenza A",
                        'Parkinson disease')
      a <- list()
      for (i in 1:4){
        a[[i]]<- gseaplot2(kk_gse, 
                           geneSetID = kk_gse@result$ID[kk_gse@result$Description==paper_choose[i]],
                           pvalue_table = T)}
      a
    }
    #gseGO
    {
      #gsemf <- gseGO(geneList,OrgDb = org.Hs.eg.db,keyType = "ENTREZID",ont="MF") #前后对应上
      #gsemf <- gseGO(geneList,OrgDb = org.Hs.eg.db,keyType = "ENTREZID",ont="BP")
      gsemf <- gseGO(geneList,OrgDb = org.Hs.eg.db,keyType = "ENTREZID",ont="all")
      head(gsemf)
    }
    #GSEA
    {
      hallmarks <- read.gmt("E:\\zhaoxt_workplace\\mywork\\code\\R\\R\\GSEA\\GSEA\\h.all.v6.2.entrez.gmt")
      
      y <- GSEA(geneList,TERM2GENE =hallmarks)
      
      library(ggplot2)
      dotplot(y,showCategory=12,split=".sign")+facet_grid(~.sign)
      yd <- data.frame(y)
      yd
      write.table(yd,"GSEA_gene_UMNJHU.csv")
      write.table(yd,"GSEA_exon_train.csv")
      #plot
      library(enrichplot)
      gseaplot2(y,"HALLMARK_HEME_METABOLISM",color = "red",pvalue_table = T)
      gseaplot2(y,"HALLMARK_INTERFERON_ALPHA_RESPONSE",color = "red",pvalue_table = T)
      gseaplot2(y,"HALLMARK_INTERFERON_GAMMA_RESPONSE",color = "red",pvalue_table = T)
    }
    #enrichPathway
    {
      library(ReactomePA)
      geneList <- df$logFC
      
      names(geneList) = df$ENTREZID
      geneList=sort(geneList,decreasing = T)
      de <- names(geneList)[abs(geneList) > logFC_cutoff]
      head(de)
      x <- enrichPathway(gene=de,pvalueCutoff=0.95, readable=T)
      head(as.data.frame(x))
      write.table(as.data.frame(x),"enrichPathway_gene_UMNJHU.csv")
      write.table(as.data.frame(x),"enrichPathway_exon_UMNJHU.csv")
      #plot
      dotplot(x, showCategory=15)
      barplot(x, showCategory=8)
      emapplot(x)
      heatplot(x)
      heatplot(x, foldChange=geneList)
      cnetplot(x, categorySize="pvalue", foldChange=geneList)
      upsetplot(x)
    }
    #gsePathway
    {
      y <- gsePathway(geneList, nPerm=100, pvalueCutoff=0.95, pAdjustMethod="BH", verbose=FALSE)
      res <- as.data.frame(y)
      head(res)
      write.table(res,"gsePathway_gene_UMNJHU.csv")
      write.table(res,"gsePathway_exon_UMNJHU.csv")
      #plot
      emapplot(y, color="pvalue")
      gseaplot(y, geneSetID = "R-HSA-194315")
      viewPathway("Hemostasis", readable=TRUE, foldChange=geneList)
    }
    #enrichKEGG
    {
      library(org.Hs.eg.db)
      library(clusterProfiler)
      library(plyr)
      kegg_SYMBOL_hsa <- function(genes){ 
        gene.df <- bitr(genes, fromType = "SYMBOL",
                        toType = c("SYMBOL", "ENTREZID"),
                        OrgDb = org.Hs.eg.db)
        head(gene.df) 
        diff.kk <- enrichKEGG(gene         = gene.df$ENTREZID,
                              organism     = 'hsa',
                              pvalueCutoff = 0.95,
                              qvalueCutoff = 0.95
        )
        return( setReadable(diff.kk, OrgDb = org.Hs.eg.db,keyType = 'ENTREZID'))
      }
      trunk_kk=kegg_SYMBOL_hsa((data$Gene))
      trunk_df=trunk_kk@result
      head(trunk_df)
      write.csv(trunk_df,file = 'enrichKEGG_gene_UMNJHU.csv')
      write.csv(trunk_df,file = 'enrichKEGG_exon_UMNJHU.csv')
      #plot
      #png(paste0('venn_probe_info_kegg_barplot', '.pdf'),width = 1080,height = 540)
      barplot(trunk_kk,font.size = 20)
      #dev.off()
    }
    #enrichGO
    {
      library(ReactomePA)
      library(DOSE)
      library(org.Hs.eg.db)
      library(clusterProfiler)
      library(clusterProfiler)
      library(topGO)
      library(Rgraphviz)
      library(RDAVIDWebService)
      library(plyr)
      library(clusterProfiler)
      
      library(stringr)
      library(ggplot2)
      
      bit_gene <- function(genes){
        gene.df <- bitr(genes, fromType = "SYMBOL",toType = c("SYMBOL", "ENTREZID"),OrgDb = org.Hs.eg.db)
        return(gene.df$ENTREZID)
      }
      
      bp <- enrichGO(bit_gene( as.character((data$Gene))), ont="all",OrgDb = "org.Hs.eg.db")
      bp@result
      dotplot(bp, split="ONTOLOGY")+ facet_grid(ONTOLOGY~., scale="free")
      
      as.data.frame(bp)
      write.csv(bp@result,file = 'enrichGO_gene_UMNJHU.csv')
      write.csv(bp@result,file = 'enrichGO_exon_UMNJHU.csv')
      #plot
      #enrichMap(bp)
      #simplify
      bp2 <-simplify(bp, cutoff=0.05,by="p.adjust", select_fun=min)
      as.data.frame(bp2)
      write.csv(bp2,file = 'enrichGO_simp_gene.csv')
      write.csv(bp2,file = 'enrichGO_simp_exon.csv')
      
      
    }
    #enrichDO
    {
      bit_gene <- function(genes){
        gene.df <- bitr(genes, fromType = "SYMBOL",toType = c("SYMBOL", "ENTREZID"),OrgDb = org.Hs.eg.db)
        return(gene.df$ENTREZID)
      }
      x <- enrichDO(gene          = bit_gene( as.character((data$Gene))), 
                    ont           = "DO",
                    pvalueCutoff  = 0.5,
                    pAdjustMethod = "BH",
                    universe      = bit_gene( as.character(colnames(all_gene)[-c(1:2)])),
                    minGSSize     = 5,
                    maxGSSize     = 500,
                    qvalueCutoff  = 0.5,
                    readable      = FALSE)
      x@result
      write.csv(x@result,file = 'enrichDO_gene_UMNJHU.csv')
      write.csv(x@result,file = 'enrichDO_exon_UMNJHU.csv')
      x <- data.frame(x)
      y <- mutate(x, richFactor = Count / as.numeric(sub("/\\d+", "", BgRatio)))
      library(ggplot2)
      library(forcats)
      library(enrichplot)
      tmp <- y[1:20,]
      ggplot(tmp, showCategory = 20,aes(richFactor, fct_reorder(Description, richFactor))) + 
        geom_segment(aes(xend=0, yend = Description)) +
        geom_point(aes(color=p.adjust, size = Count)) +
        scale_color_viridis_c(guide=guide_colorbar(reverse=TRUE)) +
        scale_size_continuous(range=c(2, 10)) +
        theme_minimal() + 
        xlab("rich factor") +
        ylab(NULL) + 
        ggtitle("Enriched Disease Ontology")
      
      #plot
      emapplot(x)
      x <- setReadable(x, 'org.Hs.eg.db')
      head(x)
      cnetplot(x)
      write.csv(x,file = 'enrichDO2_gene.csv')
      write.csv(x,file = 'enrichDO2_exon.csv')
    }
    #enrichNCG
    {
      bit_gene <- function(genes){
        gene.df <- bitr(genes, fromType = "SYMBOL",toType = c("SYMBOL", "ENTREZID"),OrgDb = org.Hs.eg.db)
        return(gene.df$ENTREZID)
      }
      ncg <- enrichNCG(bit_gene( as.character((data$Gene))),pvalueCutoff = 0.95)
      head(ncg,10)
      dotplot(ncg, showCategory=30) + ggtitle("dotplot for GSEA")
      ridgeplot(ncg)
      gseaplot2(ncg, geneSetID = 1, title = ncg$Description[1])
      gseaplot2(ncg, geneSetID = 1:3)
      gseaplot2(ncg, geneSetID = 1:3, pvalue_table = TRUE,
                color = c("#E495A5", "#86B875", "#7DB0DD"), ES_geom = "dot")
      write.csv(ncg,file = 'enrichNCG_gene.csv')
      write.csv(ncg,file = 'enrichNCG_exon.csv')
    }
    #enrichDGN
    {
      bit_gene <- function(genes){
        gene.df <- bitr(genes, fromType = "SYMBOL",toType = c("SYMBOL", "ENTREZID"),OrgDb = org.Hs.eg.db)
        return(gene.df$ENTREZID)
      }
      
      dgn <- enrichDGN(bit_gene( as.character((data$Gene))),pvalueCutoff = 0.95)
      head(dgn,20)
      write.csv(dgn,file = 'enrichDGN_gene_UMNJHU.csv')
      write.csv(dgn,file = 'enrichDGN_exon_UMNJHU.csv')
      
      #plot
      emapplot(dgn)
      edox <- setReadable(dgn, 'org.Hs.eg.db', 'ENTREZID')
      cnetplot(edox)
      
    }
  }
}

#mirna
{
  setwd("H:/dbgap_CHD/ChildStudyConsentSet_phs000363.Framingham.v16.p10.c1.HMB-IRB-MDS/ExpressionFiles/phe000005.v5.FHS_SABRe_project4_miRNA.expression-data-matrixfmt.c1/l_mrna_2011_m_0797s_13_c1.dat")
  mirna1 <- read.table("l_mrna_2011_m_0797s_13_c1.dat", sep="\t",header = T)
  mirna2 <- read.table("l_mrna_2011_m_0797s_13_c2.dat", sep="\t",header = T)
  mirna <- rbind(mirna1,mirna2)
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  load("UMN_DMP_new.Rdata")
  load("test/JHU_DMP_new.Rdata")
  
  UMN_meta_new <- rownames_to_column(UMN_DMP_new,"shareid")
  UMN_meta_new$shareid <-  gsub('X','',UMN_meta_new$shareid)
  JHU_meta_new <- rownames_to_column(JHU_DMP_new,"shareid")
  
  UMN_mirna <- merge(UMN_meta_new[,c("shareid","chf")],mirna,by="shareid")#672
  JHU_mirna <- merge(JHU_meta_new[,c("shareid","chf")],mirna,by="shareid")#163
  
  mirna <- rbind(UMN_mirna,JHU_mirna)
  mirna <- data.frame(mirna)
  mirna <- mirna[,-c(386,387,384)]
  save(mirna,file="mirna.Rdata")
  write.table(mirna,"mirna.csv",sep=",",row.names = T)
  boxplot(mirna[,-c(1:2)])
  
  library(limma)
  mirna2 = log2(mirna[,-c(1,2)]+1)
  boxplot(mirna2)
  #mirna2 = normalizeBetweenArrays(mirna[,3:414])
  # DEG
  {
    
    load(file="../RNA/mirna.Rdata")
    mirna[1:4,1:4]
    # rm NA
    a = colSums(is.na(mirna[,-c(1,2)])) > nrow(mirna)*0.2
    b = mirna[,-1][,a]
    b[1:4,1:4]
    #boxplot(b[,-1])
    #boxplot(log(b[,-1]+1))
    #mirna2 = normalizeBetweenArrays(b)
    #b2 = log(b[,-1]+1)
    
    library(limma)
    
    group_list=factor(b$chf)
    b = b[,-1]
    n_expr = b[grep( "1", group_list ),]
    g_expr = b[grep( "0", group_list ),]
    exprSet = rbind( n_expr, g_expr )
    exprSet = t(exprSet)
    exprSet[1:4,1:4]
    group_list = c(rep( 'chf', nrow( n_expr ) ),rep( 'nochf',    nrow( g_expr ) ) )
    design <- model.matrix(~0 + factor(group_list)) 
    colnames(design) = levels(factor(group_list)) 
    rownames(design) = colnames(exprSet)   ## 
    design  
    
    contrast.matrix <- makeContrasts(paste0(unique(group_list),collapse = "-"), levels = design) 
    contrast.matrix  
    
    ####### 做DEG差异分析
    exprSet[1:4,1:4]
    fit <- lmFit(exprSet, design)   
    fit2 <- contrasts.fit(fit, contrast.matrix) 
    fit2 <- eBayes( fit2 )   ## Empirical Bayes Statistics for Differential Expression
    nrDEG = topTable( fit2, coef = 1, n = Inf )#231
    write.table( nrDEG,  "mirna_DEG.csv",sep=",")
    head(nrDEG)
    nrDEG$ID <- rownames(nrDEG)
    nrDEG_ <- filter(nrDEG,P.Value != "NA")
    summary(nrDEG_$logFC)
    logFC_cutoff <- with( nrDEG_, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff#3.730608
    #data <- nrDEG_[abs(nrDEG_$logFC) > logFC_cutoff,]
    data <- nrDEG_[abs(nrDEG_$logFC) > 1,]
    data$ID
    write.table( data,  "mirna_DEG_diff_train.csv",sep=",")
    #only train set
    #"miR_210"     "miR_192_5p"  "miR_146a_5p" "miR_141_5p"  "miR_483_5p"  "miR_452_5p"  "miR_449b_5p"
    #hsa-miR-210;hsa-miR-192-5p;hsa-miR-146a-5p;hsa-miR-141-5p;hsa-miR-483-5p;hsa-miR-452-5p;hsa-miR-449b-5p    
    #train and test set
    #"miR_503"      "miR_154_"     "miR_29b_1_5p"
    #hsa-miR-503;hsa-miR-154;hsa-miR-29b-1-5p;
  }
}

#pair cpg and RNA(train)
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  #gene or exon
  {
    result = result1#gene,818 26
    result = result2#exon,818 26
  }
  #cpg
  {
    id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
    head(id3)
    id3 <- as.character(id3$Feature)
    id3[-c(2,6,17,20,21)]
    cpg_1 <- train[colnames(train) %in% c("shareid",id3[-c(2,6,17,20,21)])]
    cpg_2 <- test[colnames(test) %in% c("shareid",id3[-c(2,6,17,20,21)])]
    cpg = rbind(cpg_1,cpg_2)
    cpg = cpg_1
    cpg <- cpg[cpg$shareid %in% result$shareid,]#818
  }
  #cgp and ehr
  {
    data_train <- train[colnames(train) %in% c("shareid",id3)]
    data_test <- test[colnames(test) %in% c("shareid",id3)]
    Da = rbind(data_train,data_test)
    {
      cormat<-round(cor(Patient_impute,method = "spearman"),2)
      cormat[1:4,1:4]
      row_name <- ""
      col_name <- ""
      n <- 1
      for (i in 1:nrow(cormat)) {
        for (ii in 1:ncol(cormat)) {
          if (cormat[i,ii] != 1 & abs(cormat[i,ii]) >= 0.8 ){
            row_name[n] <- i
            col_name[n] <- ii
            n=n+1
          }
          
        }
      }
      row_name=as.numeric(row_name)
      col_name=as.numeric(col_name)
      cg_cor_data=cbind.data.frame(row_name,col_name)
      cg_cor_data
      colnames(cormat)
    }
    cor.test(as.numeric(Da$Age),as.numeric(Da$Diuretic),method="pearson")
  }
  #pair RNA & CpG
  {
    dim(cpg);dim(result)
    data <- merge(cpg,result,by="shareid")
    data[1:10,1:10];colnames(data)
    data$Heart.failure <- ifelse(data$Heart.failure == "0", "No-Heart.failure","Heart.failure")
    colnames(data)[27] = "Sample" #chf.x
    methylation = read_excel("E:\\workplace\\mywork\\1-paper\\0.ppt_word\\9 clincal and translational medicine\\venn_probe_The 25 CpGs associated with HFrisk model.xlsx",sheet=1, na = "NA")
    venn_CpG = methylation[methylation$Probe %in% c(id3),] #22 cpg+cor_cpg
    tmp = venn_CpG[-24,]#missing: "MRI1"
    #cor
    {
      method = "pearson"
      cormat<-round(cor(data[data$Sample == "Heart.failure",-c(1,27)],method = method),2)
      row_name <- ""
      col_name <- ""
      n <- 1
      for (i in 1:nrow(cormat)) {
        for (ii in 1:ncol(cormat)) {
          if (cormat[i,ii] != 1 & abs(cormat[i,ii]) >= 0.3 ){
            row_name[n] <- i
            col_name[n] <- ii
            n=n+1}}}
      row_name=as.numeric(row_name)
      col_name=as.numeric(col_name)
      cg_cor_data=cbind.data.frame(row_name,col_name)
      cg_cor_data
      cg_cor_data = cg_cor_data[cg_cor_data$row_name < 25 & cg_cor_data$col_name >25,]
      colnames(cormat)
      cg_cor_data$cor = ""
      cg_cor_data$cpg = ""
      cg_cor_data$rna = ""
      for(j in 1:nrow(cg_cor_data)){
        cg_cor_data[j,3] = cormat[cg_cor_data$row_name[j],cg_cor_data$col_name[j]]
        cg_cor_data[j,4] = colnames(cormat)[cg_cor_data$row_name[j]]
        cg_cor_data[j,5] = colnames(cormat)[cg_cor_data$col_name[j]]
      }
      all_gene_d
      all_exon_d
      train_gene_d
      train_exon_d
    }
    #cor.test
    {
      cor1=""
      p1=""
      method = "pearson"
      for(i in 2:26){
        for(m in 28:51){
          cor=cor.test(as.numeric(data[data$Sample == "Heart.failure",i]),as.numeric(data[data$Sample == "Heart.failure",m]),method=method)[[4]]
          p=cor.test(as.numeric(data[data$Sample == "Heart.failure",i]),as.numeric(data[data$Sample == "Heart.failure",m]),method=method)[[3]]
          cor=data.frame(cor)
          p=data.frame(p)
          colnames(cor)= paste(colnames(data)[i],colnames(data)[m],sep = "_")
          colnames(p)= paste(colnames(data)[i],colnames(data)[m],sep = "_")
          cor1=cbind(cor1,cor)
          p1=cbind(p1,p)
        }
      }
      cor1 = cor1[,-1]
      p1 = p1[,-1]
      c_r = rbind(cor1,p1)
      rownames(c_r) = c("cor","pvalue")
      c_r = as.data.frame(t(c_r))
    }
  }
  #all RNA & CpG
  {
    dim(cpg);dim(result)
    data <- merge(cpg,result,by="shareid")
    data[1:10,1:10];colnames(data)
    data$Heart.failure <- ifelse(data$Heart.failure == "0", "No-Heart.failure","Heart.failure")
    colnames(data)[27] = "Sample" #chf.x
    #cor
    {
      cormat<-round(cor(data[data$Sample == "Heart.failure",-c(1,27)],method = "spearman"),2)
      row_name <- ""
      col_name <- ""
      n <- 1
      for (i in 1:nrow(cormat)) {
        for (ii in 1:ncol(cormat)) {
          if (cormat[i,ii] != 1 & abs(cormat[i,ii]) >= 0.4 ){
            row_name[n] <- i
            col_name[n] <- ii
            n=n+1}}}
      row_name=as.numeric(row_name)
      col_name=as.numeric(col_name)
      cg_cor_data=cbind.data.frame(row_name,col_name)
      cg_cor_data
      cg_cor_data = cg_cor_data[cg_cor_data$row_name < 25 & cg_cor_data$col_name >25,]
      colnames(cormat)
      cg_cor_data$cor = ""
      cg_cor_data$cpg = ""
      cg_cor_data$rna = ""
      for(j in 1:nrow(cg_cor_data)){
        cg_cor_data[j,3] = cormat[cg_cor_data$row_name[j],cg_cor_data$col_name[j]]
        cg_cor_data[j,4] = colnames(cormat)[cg_cor_data$row_name[j]]
        cg_cor_data[j,5] = colnames(cormat)[cg_cor_data$col_name[j]]
      }
    }
  }
  #ggplot
  {
    library(ggpubr)
    library(cowplot)
    
    plot_result <- lapply(c(tmp$`Closest gene`), function(x){
      cpg <- as.character(tmp[tmp$`Closest gene` == x,1])
      pv <- data.frame(CpG = data[,cpg], Gene = data[,x],Sample = data$Sample)# "" is not need
      lab <- paste(cpg,"and", x,sep=" ")
      
      plot_grid(ggplot(pv, aes(x = CpG, y = Gene)) +
                  geom_point(aes(color = Sample, shape = Sample)) +
                  geom_rug(aes(color =Sample)) +
                  geom_smooth(aes(color = Sample, fill = Sample), method = lm)+
                  scale_color_manual(values = c("#00AFBB", "#E7B800"))+
                  scale_fill_manual(values = c("#00AFBB", "#E7B800"))+
                  ggtitle(lab) +
                  ggpubr::stat_cor(aes(color = Sample), method = "pearson")
      )
    })
    plot_grid(plotlist=plot_result, ncol=6)
    ggsave(file="new_champ\\ggplot_Cpg-RNA_gene.pdf",width=45.5, height=25.6)
    ggsave(file="new_champ\\ggplot_Cpg-RNA_gene_train.pdf",width=45.5, height=25.6)
    ggsave(file="new_champ\\ggplot_Cpg-RNA_exon.pdf",width=45.5, height=25.6)
    ggsave(file="new_champ\\ggplot_Cpg-RNA_exon_train.pdf",width=45.5, height=25.6)
  }
  #ggscatterhist
  {
    # Use box plot as marginal plots
    library(ggpubr)
    library(cowplot)
    
    plot_result <- lapply(c(tmp$`Closest gene`), function(x){
      cpg <- as.character(tmp[tmp$`Closest gene` == x,1])
      #pv <- data.frame(cpg = data[,cpg], gene = data[,x],Sample = data$chf.x)#"" is need 
      plot_grid(ggscatterhist(
        data, x = cpg, y = x,
        color = "Sample", size = 3, alpha = 0.6,
        palette = c("#00AFBB", "#E7B800"),
        margin.params = list(fill = "Sample", color = "black", size = 0.2),
        margin.plot = "boxplot",
        ggtheme = theme_bw()
      ))
    })
    plot_grid(plotlist=plot_result, ncol=6)
    ggsave(file="new_champ/ggscatterhist_Cpg-RNA_gene.pdf",width=45.5, height=25.6)
    ggsave(file="new_champ/ggscatterhist_Cpg-RNA_exon.pdf",width=45.5, height=25.6)
  }
  #ggscatterstats
  {
    library(cowplot)
    library(ggstatsplot)
    plot_result <- lapply(c(tmp$`Closest gene`), function(x){
      cpg <- as.character(tmp[tmp$`Closest gene` == x,1])
      pv <- data.frame(cpg = data[,cpg], gene = data[,x],Sample = data$Sample)#"" is need 
      lab <- paste(cpg,"and", x,sep=" ")
      plot_grid(ggscatterstats(data = pv,
                               x = cpg,
                               y = gene,
                               type="nonparametric",
                               title = lab,
                               marginal.type = "boxplot"))
    })
    plot_grid(plotlist=plot_result, ncol=6)
    ggsave(file="ggscatterstats_Cpg-RNA_gene.pdf",width=45.5, height=25.6)
    ggsave(file="ggscatterstats_Cpg-RNA_exon.pdf",width=45.5, height=25.6)
  }
}

#DEG & DMG
{
  #gene
  {
    nrDEG = read.table( "./all_gene_DEG_UMNJHU.csv",sep=",")
    head(nrDEG)
    nrDEG = rownames_to_column(nrDEG,"ID")
    nrDEG_ <- filter(nrDEG,P.Value < 0.05)
    summary(nrDEG_$logFC)
    logFC_cutoff <- with( nrDEG, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff#0.1469162
    data1 <- nrDEG[abs(nrDEG$logFC) > logFC_cutoff,]
    data1 <- data1[data1$P.Value < 0.05,] #449
    head(data1)
  }
  #exon
  {
    nrDEG = read.table( "./all_exon_DEG_UMNJHU.csv",sep=",")
    head(nrDEG)
    nrDEG = rownames_to_column(nrDEG,"ID")
    nrDEG_ <- filter(nrDEG,P.Value < 0.05)
    summary(nrDEG_$logFC)
    logFC_cutoff <- with( nrDEG, mean( abs( logFC ) ) + 2 * sd( abs( logFC ) ) )
    logFC_cutoff #0.202579
    data2 <- nrDEG[abs(nrDEG$logFC) > logFC_cutoff,]
    data2 <- data2[data2$P.Value < 0.05,] #491
  }
  #cpg
  cpg_gene <- c("GRIK4","MRI1","RHOBTB1","GARS","CYP2E1","KCNIP4","CDH4","GRM4" ,
                "VIPR2","CACNA1H" ,"SLC25A2","GDF7" , "FBXO28","ZBTB20" ,"SLC1A4",
                "ITGB1BP1","DLGAP1")
  overlap1 = intersect(data1$ID,cpg_gene)#empty
  overlap2 = intersect(data2$ID,cpg_gene)#empty
}