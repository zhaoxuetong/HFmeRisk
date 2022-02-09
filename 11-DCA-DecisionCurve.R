#https://mp.weixin.qq.com/s?__biz=MzA5ODQ1NDIyMQ==&mid=2649607711&idx=2&sn=9e9e69e9f6cb468c56bbf6e989a46f2d&chksm=888831bdbfffb8ab9d3b7d1509d2ae1fae041e0b816e92c00c015775d3356c2f357cd76451ef&mpshare=1&scene=1&srcid=0822NspPgV0JaSJvwe1YhCZZ#rd
library(rmda)
##library(DecisionCurve) 
setwd("E:/workplace/mywork/methy/dbgap/chf/data_chf_contr/early_chf/c1_UMN_JHU/train_UMN_tset_JHU/1123_dataSummary/")
{
  load(file="test//test_data.Rdata")
  id3 <- read.table("xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  id3[2] = "Age"
  id3[6] = "Diuretic"
  id3[17] = "BMI"
  id3[20] = "Creatinine.serum"
  id3[21] = "Albumin.urine"
  #JHU_DMP_new <- rownames_to_column(JHU_DMP_new,"id")
  #data <- JHU_DMP_new[,c("id","chf",id3)]
  
  #paste(id3[c(2,6,17,20,21)], collapse = "+")
  
  Data0 <- test_data[,c(id3[c(2,6,17,20,21)],"Heart.failure")]
  Data1 <- test_data[,c(id3[-c(2,6,17,20,21)],"Heart.failure")]
  Data2 <- test_data[,c(id3,"Heart.failure")]
  simple<- decision_curve(Heart.failure~Age+Diuretic+BMI+Creatinine.serum+Albumin.urine+Heart.failure,
                          data = Data0, family = binomial(link ='logit'),
                          thresholds= seq(0,1, by = 0.01),
                          confidence.intervals =0.95,study.design = 'case-control',
                          population.prevalence = 0.3)
  paste(colnames(Data1),collapse = "+")
  
  simple2<- decision_curve(Heart.failure~cg20051875+cg17766026+cg10556349+cg00495303+cg03233656+cg25755428+cg05845376+cg10083824+cg05481257+cg08614290+cg24205914+cg03556243+cg08101977+cg13352914+cg05363438+cg21429551+cg07041999+cg27401945+cg11853697+cg21024264+cg06344265+cg00522231+cg16781992+cg00045910+cg23299445,
                           data = Data1, family = binomial(link ='logit'),
                          thresholds= seq(0,1, by = 0.01),
                          confidence.intervals =0.95,study.design = 'case-control',
                          population.prevalence = 0.3)
  
  paste(colnames(Data2),collapse = "+")
  complex<- decision_curve(Heart.failure~cg20051875+Age+cg17766026+cg10556349+cg00495303+Diuretic+cg03233656+cg25755428+cg05845376+cg10083824+cg05481257+cg08614290+cg24205914+cg03556243+cg08101977+cg13352914+BMI+cg05363438+cg21429551+Creatinine.serum+Albumin.urine+cg07041999+cg27401945+cg11853697+cg21024264+cg06344265+cg00522231+cg16781992+cg00045910+cg23299445,
                           data = Data2,
                           family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01),
                           confidence.intervals= 0.95,study.design = 'case-control',
                           population.prevalence= 0.3)
  List<- list(complex,simple,simple2)
  plot_decision_curve(List,curve.names= c('HFrisk',"5 EHR",'25 CpG'),
                      cost.benefit.axis =F,col = c("#F8766D","#00AFBB","#E7B800"),
                      confidence.intervals =F,standardize = T)+
    annotate(geom = "text", x = 20, y = 2.8, label = "Times New Roman", size = 12, color = "tomato2", family = "newman")
  plot_clinical_impact(complex,population.size = 1000,cost.benefit.axis = T,
                       n.cost.benefits= 8,col = c("#F8766D","black"),
                       confidence.intervals= F)
}