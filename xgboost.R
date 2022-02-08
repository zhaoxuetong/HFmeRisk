# For the DMP results obtained from the UMN dataset, perform lasso feature selection and then perform xgboost analysis

require(xgboost)
library(caret)

#dmp lasso xgboost
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  id <- read.table("lasso_dmp_little.csv")
  load("UMN_DMP_new.Rdata")
  head(id)
  id <- as.character(id$V1)
  id <- id[-1]
  
  X <- UMN_DMP_new[,colnames(UMN_DMP_new) %in% id]
  X <- subset(X, select=id)
  y <- as.factor(UMN_DMP_new$chf)
  
  set.seed(1234)
  samp=sample(1:nrow(X),round(nrow(X)/6))	
  X_test=X[samp,]
  y_test=y[samp]
  X=X[-samp,]
  y=y[-samp]
  
  set.seed(1234)
  
  grid = expand.grid(
    nrounds = c(5:15),
    colsample_bytree = c(0.2,0.5,0.8,1), #1
    min_child_weight = c(1,2,3), #1
    eta = c(0.5,0.3, 0.8, 1 ), #0.3 is default
    gamma = c(0.5, 0.25,0.1,0.15), #0
    subsample = c(0.5,0.75,0.6,0.4), #1
    max_depth = c(2,3,4,6) #6
  )
  
  library(caret)
  cntrl = trainControl(method = "cv",number = 5,verboseIter = TRUE, returnData = FALSE,returnResamp = "final")
  
  set.seed(1234)
  train.xgb = train(x = X, y = y,trControl = cntrl,tuneGrid = grid,method = "xgbTree")
  train.xgb
  #Fitting nrounds = 12, max_depth = 4, eta = 0.5, gamma = 0.1, colsample_bytree = 0.8, 
  #min_child_weight = 2, subsample = 0.5 on full training set
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "error",
                  nrounds             = 12,
                  eta                 = 0.5, 
                  max_depth           = 4, 
                  subsample           = 0.5,
                  colsample_bytree    = 0.8,
                  min_child_weight    = 2 ,
                  gamma               = 0.1
  )
  x <- as.matrix(X)
  y <- as.numeric(UMN_DMP_new$chf[-samp])
  train.mat <- xgb.DMatrix(data = x, label = y)
  set.seed(1234)
  xgb.fit <- xgb.train(params = param, data = train.mat, nrounds = 8)
  xgb.fit
  
  impMatrix <- xgb.importance(feature_names = colnames(X), model = xgb.fit)
  impMatrix #28,27
  write.table(impMatrix,"xgblasso_DMP.csv",sep=",",row.names = F)
}
