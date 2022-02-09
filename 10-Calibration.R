library(rms)
library(tibble)
library(rio)  
library(rms)
library(tibble)
library(classifierplots)
library(PredictABEL)
library(ggplot2)
library(cowplot)
library(riskRegression) 
library(rio)
library(rms)
library(tibble)
library(dplyr)

#data
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/test")
  load(file="test_data.Rdata")
  id3 <- read.table("../xgblasso_DMP.csv",sep=",",header = T)
  head(id3)
  id3 <- as.character(id3$Feature)
  id3[2] = "Age"
  id3[6] = "Diuretic"
  id3[17] = "BMI"
  id3[20] = "Creatinine.serum"
  id3[21] = "Albumin.urine"
  test_data <- rownames_to_column(test_data,"id")
  data <- test_data[,colnames(test_data) %in% c("id","Heart.failure",id3)]
  data[1:10,1:10]
  #data$id <-  gsub('X','',data$id)
  colnames(data)
  #rmid <- c("12049","1863","205","7249","8517","9644")
  #data <- data[!data$id %in% rmid,]
  
  load(file="test_meta_raw.Rdata")
  test_meta_raw[1:10,1:10]
  tmp <- merge(data,test_meta_raw[,c("Sample_Name","chfdate","DATE8")],by.x="id",by.y="Sample_Name")
  tmp <- tmp[!duplicated(tmp$id),]
  tmp$survTime <- tmp$chfdate - tmp$DATE8
}

#plot1
{
  #paste(id3, collapse = "+")
  s <- Surv(tmp$survTime,tmp$"Heart.failure",type="right")
  f <- cph(s ~ cg20051875+Age+cg17766026+cg10556349+cg00495303+
             Diuretic+cg03233656+cg25755428+cg05845376+
             cg10083824+cg05481257+cg08614290+cg24205914+
             cg03556243+cg08101977+cg13352914+BMI+
             cg05363438+cg21429551+Creatinine.serum+Albumin.urine+
             cg07041999+cg27401945+cg11853697+cg21024264+
             cg06344265+cg00522231+cg16781992+cg00045910+cg23299445,
           x=TRUE, y=TRUE,surv = TRUE,time.inc=3000,data=tmp)
  
  cal <- calibrate(f,u=3000,cmethod='KM',m=45)
  plot(cal,xlim = c(0,1),ylim= c(0,1), errbar.col=c(rgb(0,0,0,maxColorValue=255)),col=c(rgb(255,0,0,maxColorValue=255)))
  abline(0,1,lty=3,lwd=2,col=c(rgb(0,0,255,maxColorValue= 255)))
}
#plotCalibration
{
  #deepfm test result
  prob <- read.table(header = T,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\new_1126\\deepFM _Mean0.80126_Std0.01731.csv",sep=",")
  #prob <- read.table(header = T,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\1126_dmp_lassoxgboost_HFrisk_Mean0.79092_Std0.00320.csv",sep=",")
  #prob <- read.table(header = T,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\old\\DeepFM_EHRCPG_Mean0.85025_Std0.05789.csv",sep=",")
  #prob <- read.table(header = T,"D:\\anaconda-python\\learn_DL\\Basic-DeepFM-model\\output\\re_HFrisk_nolassoxgboost_CPG_Mean0.25721_Std0.00333.csv",sep=",")
  dat = read.csv("../champ_rawdata_test.csv",header = T,skip = 7)
  dat = merge(dat[,c(1,12)],prob,by.x="shareid",by.y="ID")
  data = merge(dat[,-1],data,by.x="Sample_Name",by.y="id")
  colnames(data)
  classifierplots_folder(data$Heart.failure,data$target,folder = "E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/calibrate",height = 100,width = 100)
  val.prob(data$target,as.numeric( data$Heart.failure) , m=10, cex=.9) 
}
#plot2
{
  g = 10
  #install.packages("ResourceSelection")
  library(ResourceSelection)
  data$Heart.failure <- as.numeric(data$Heart.failure ) ##attention
  HL <- hoslem.test(data$Heart.failure,data$target, g=g)#g=10
  HL#X-squared = 6.8495, df = 8, p-value = 0.553
  cbind(HL$observed,HL$expected)
  prob <- data$target
  HL1 <- plotCalibration(data,4, prob, groups=g)  
  HL1
  plotCalibration(data,4, prob, groups=g)
  
  # $Chi_square
  # [1] 6.839 
  # $df
  # [1] 8
  # $p_value
  # [1] 0.5541
  HLdata<- data.frame(HL1$Table_HLtest)
  pv1.lab <- paste("Chi square =", HL1$Chi_square)
  pv2.lab <- paste("p value =", HL1$p_value)
  pv3.lab <- paste("Hosmer-Lemeshow p =", round(HL$p.value, 3))
  title <- paste("Calibration plot",pv1.lab,pv3.lab, sep="\n")
  title <- paste("Calibration plot",pv3.lab, sep="\n")
  lab <- paste(pv1.lab, pv2.lab, sep="\n")
  label <- ggdraw() + draw_label(lab)
  
  #plot_grid(label, 
  ggplot(HLdata,aes(x=meanpred,y=meanobs))+
    geom_point(shape=17,size=3,color="blue")+
    #geom_text(aes(x=meanpred, y= meanobs,label=paste("p value= ",round(HL1$p_value, 2))),data = HLdata,inherit.aes=F)+
    geom_abline(intercept=0,slope=1,color="black",linetype=2)+
    labs(x="Predicted risk",y="Observed risk",title=title)+
    xlim(0,1)+ylim(0,1)+theme_bw()+
    theme(plot.title = element_text(hjust = 0.5))+
    theme(panel.grid =element_blank()) + 
    #theme(panel.border = element_blank()) +
    theme(axis.line = element_line(colour = "black"))
  #,ncol=1, rel_heights=c(.3, 1))
}
#plot3
{
  data <- test_data[,c("Heart.failure",id3)]
  m1 <- glm(Heart.failure~., data = data,family = binomial(link="logit"))
  m2 <- glm(Heart.failure~Age+Diuretic+BMI+Creatinine.serum+Albumin.urine,data = data, family = binomial(link="logit"))
  m3 <- glm(Heart.failure~cg20051875+cg17766026+cg10556349+cg00495303+
              cg03233656+cg25755428+cg05845376+cg10083824+cg05481257+
              cg08614290+cg24205914+cg03556243+cg08101977+cg13352914+
              cg05363438+cg21429551+cg07041999+cg27401945+cg11853697+
              cg21024264+cg06344265+cg00522231+cg16781992+cg00045910+
              cg23299445,data = data, family = binomial(link="logit"))
  jz=Score(list(model1=m1,model2=m2,model3=m3),Heart.failure~1,data=data,plots="calibration")
  plotCalibration(jz,col=c("black","red","blue"),lty=c(1,4))
  plotCalibration(jz,models=1,bars=TRUE,show.frequencies=T,names.cex=0.8,col=c("red","blue"))
  m3 <- lrm(Heart.failure~., x = T,y = T,data = data)
  cal <- calibrate(m3,  method = "boot", B = 1000)
  plot(cal,main = "Calibration Curve")
  newdata <- data
  pred.lg<- predict(m3,newdata )
  newdata$prob <- 1/(1+exp(-pred.lg))
  val.prob(newdata$prob,as.numeric( newdata$Heart.failure) , m=10, cex=.9)
}
