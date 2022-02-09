rm(list=ls())
library(tibble)
library(dplyr)
library(glmnet)
library(lars) 
library(VennDiagram)
library(sigFeature)
library(e1071)
library(caret)
library(randomForest)
library(ltm)
#UMN_DMP_new：dmp lasso
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  load("UMN_DMP_new.Rdata")
  X <- as.matrix(UMN_DMP_new[,!colnames(UMN_DMP_new) %in% c("chf","cvd")]) #chf
  y <- as.character(UMN_DMP_new$chf)
  set.seed(2)
  fitCV <- cv.glmnet(X, y, family = "binomial", type.measure = "auc", nfolds = 10)
  #lambda.min
  myCoefs <- coef(fitCV, s = "lambda.min")
  lasso_fea <- myCoefs@Dimnames[[1]][which(myCoefs[,1] != 0 )]
  lasso_fea <- data.frame(lasso_fea)
  lasso_fea$x <- myCoefs@x 
  lasso_fea <- lasso_fea[-1,]
  length(lasso_fea[,1])#103
  
  #lambda.1se
  myCoefs2 <- coef(fitCV, s="lambda.1se")
  #lambda.1se
  lasso_fea2 <- myCoefs2@Dimnames[[1]][which(myCoefs2[,1] != 0 )]
  lasso_fea2 <- data.frame(lasso_fea2)
  lasso_fea2$x <- myCoefs2@x 
  lasso_fea2 <- lasso_fea2[-1,]
  length(lasso_fea2[,1])#91
  
  set.seed(2)
  cv_fit <- cv.glmnet(X, y, nfold=10, alpha = 1, family = "binomial", type.measure = "class")
  myCoefs3 <- coef(cv_fit, s="lambda.min")
  #lambda.min
  lasso_fea3 <- myCoefs3@Dimnames[[1]][which(myCoefs3[,1] != 0 )]
  lasso_fea3 <- data.frame(lasso_fea3)
  lasso_fea3$x <- myCoefs3@x 
  lasso_fea3 <- lasso_fea3[-1,]
  length(lasso_fea3[,1])#104
  
  #2
  myCoefs4 <- coef(cv_fit, s="lambda.1se")
  #lambda.1se
  lasso_fea4 <- myCoefs4@Dimnames[[1]][which(myCoefs4[,1] != 0 )]
  lasso_fea4 <- data.frame(lasso_fea4)
  lasso_fea4$x <- myCoefs4@x 
  lasso_fea4 <- lasso_fea4[-1,]
  length(lasso_fea4[,1])#92
  
  require("VennDiagram")
  #VENN.LIST=list(lasso2_lambda1se=lasso_fea4$lasso_fea4,lasso2_lambdamin=lasso_fea3$lasso_fea3,lasso1_lambda1se=lasso_fea2$lasso_fea2,lasso1_lambdamin=lasso_fea$lasso_fea)
  VENN.LIST=list(B_lambda1se=lasso_fea4$lasso_fea4,B_lambdamin=lasso_fea3$lasso_fea3,A_lambda1se=lasso_fea2$lasso_fea2,A_lambdamin=lasso_fea$lasso_fea)
  venn.plot <- venn.diagram(VENN.LIST , NULL, 
                            fill=c("darkmagenta", "darkblue","red","black"), 
                            alpha=c(0.5,0.5,0.5,0.5), cex = 2, 
                            cat.fontface="plain", cat.cex = 1.5, 
                            main="Overlap of measure in 1se and min ")
  grid.draw(venn.plot) 
  venn4=intersect(intersect(intersect(lasso_fea$lasso_fea,lasso_fea3$lasso_fea3),lasso_fea4$lasso_fea4),lasso_fea2$lasso_fea2)
  venn3=union(union(union(lasso_fea$lasso_fea,lasso_fea3$lasso_fea3),lasso_fea4$lasso_fea4),lasso_fea2$lasso_fea2)
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  write.csv(venn4,'lasso_dmp_little.csv', row.names = F)
  write.csv(venn3,'lasso_dmp_large.csv', row.names = F)
  {
    library(UpSetR)
    B_lambda1se=lasso_fea4$lasso_fea4
    B_lambdamin=lasso_fea3$lasso_fea3
    A_lambda1se=lasso_fea2$lasso_fea2
    A_lambdamin=lasso_fea$lasso_fea
    input <- fromList(list(A_lambdamin=lasso_fea$lasso_fea, 
                           A_lambda1se=lasso_fea2$lasso_fea2,
                           B_lambdamin=lasso_fea3$lasso_fea3,
                           B_lambda1se=lasso_fea4$lasso_fea4))
    upset(input, order.by = "freq",nsets = 7,point.size = 3) 
  }
}