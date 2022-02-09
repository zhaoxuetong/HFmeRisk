rm(list=ls())

#merge EHR and CpG
{
  library(impute)
  library(tibble)
  library(dplyr)
  
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
  load(file="CellFraction.Rdata")
  load(file="HFpEF.Rdata")
  HFpEF <- data.frame(HFpEF)
  data_cpg = filter(HFpEF,PACKS_SET == "UMN")
  data_cpg_chf <- filter(data_cpg , chf == 1)
  data_cpg_nochf <- filter(data_cpg , chf == 0)
  data_cpg <- rbind(data_cpg_chf,data_cpg_nochf)
  data_cpg$Sample_Name <- paste(rep(c("chf","control"),c(nrow(data_cpg_chf),nrow(data_cpg_nochf))),
                                c(c(1:nrow(data_cpg_chf)),c(1:nrow(data_cpg_nochf))),sep = "")
  train_meta  <- data_cpg[,-c(4:7)]
  train_meta <- train_meta[,c(128,1:127)]
  train_meta_raw <- cbind(train_meta,CellFraction)
  {
    colnames(train_meta_raw)= c("Sample_Name","shareid",
                                "Ejectionfraction","PACKS_SET",
                                "Omega3","Omega3amount",
                                "Statin","Statinamount",
                                "Thiazides","Thiazidesamount",
                                "Diuretic","Diureticamount",
                                "Potassium","Potassiumamount",
                                "Aldosterone","Aldosteroneamount",
                                "Amiodarone","Amiodaroneamount",
                                "Vasodilators","Vasodilatorsamount",
                                "CoQ10","CoQ10amount",
                                "Betablocking","Betablockingamount",
                                "AngiotensinIIantagonists","AngiotensinIIantagonistsamount",
                                "ACEI","ACEIamount",
                                "Warfarin","Warfarinamount",
                                "Clopidogrel","Clopidogrelamount",
                                "Aspirin","Aspirinamount",
                                "Folicacid","Folicacidamount",
                                "cvd","Coronaryheartdisease","Heartfailure",
                                "chddate","chfdate","cvddate","midate",
                                "Myocardialinfarction","Diabetes",
                                "Atrialfibrillation","afxdate",
                                "Stroke","strokedate",
                                "DATE8","DATE9",
                                "Gender","Age","Bloodglucose","BMI","LDLcholesterol",
                                "Numberofcigarettessmoked","Creatinineserum",
                                "Smoking","Averagediastolicbloodpressure",
                                "Leftventricularhypertrophy",
                                "Fastingbloodglucose","HDLcholesterol","Hight",
                                "Averagesystolicbloodpressure",
                                "Totalcholesterol","Triglycerides" ,"Ventricularrate",
                                "Waist","Weight","Treatedforhypertension",
                                "Treatedforlipids","aspirin.1",
                                "Drinkbeer","Drinkwine","Drinkliquor","Sleep", 
                                "Albuminurine", "Creatinineurine" ,
                                "HemoglobinA1cwholeblood","Atrialenlargement",
                                "Rightventricularhypertrophy","lvh","Rheumatic",
                                "Aorticvalve","Mitralvalve","other_heart",
                                "Arrhythmia","other_peripheral_vascular_disease",
                                "other_vascular_diagnosis",
                                "Dementia","Parkinson","Adultseizuredisorder" ,
                                "Neurological","Thyroid","Endocrine","Renal","Gynecologic",
                                "Emphysema","Pneumonia","Asthma","Pulmonary","Gout",
                                "Degenerative","Rheumatoidarthritis","Musculoskeletal","Gallbladder",
                                "Gerd","Liver","Gidisease","Hematologicdisorder","Bleedingdisorder",
                                "Eye","Ent","Skin","other","Depression","Anxiety","Psychosis",
                                "other2","Prostate","Infectious","Fever","pneumonia.1",
                                "Chronicbronchitis","emphysema.1","COPD","Creactiveprotein",
                                "CD8T","CD4T","NK","Bcell","Mono","Gran")
  }
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/")
  save(train_meta_raw,file="train_meta_raw.Rdata")
  train_meta = train_meta_raw[,!colnames(train_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                 "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                 "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                 "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                 "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                 "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                 "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                 "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
  
  
}

#EHR
{
  #sex
  train_meta$Gender <- ifelse(train_meta$Gender == 1,"F","M")
  train_meta = train_meta[,-c(1,2)]
  
  {
    library(table1)
    library(tableone)
    {
      #0,1 feature 
      c = c("Omega3" ,"Statin" , "Thiazides" ,"Diuretic" ,"Potassium" ,"Aldosterone",
            "Amiodarone","Vasodilators" , "CoQ10", "Betablocking",
            "AngiotensinIIantagonists" , "ACEI", "Warfarin","Clopidogrel" ,"Aspirin" , 
            "Folicacid",
            "Coronaryheartdisease", "Myocardialinfarction", "Diabetes","Atrialfibrillation" , "Stroke" ,
            "Smoking" ,"Leftventricularhypertrophy" ,"Treatedforhypertension", "Treatedforlipids" ,
            "Drinkbeer", "Drinkwine" ,"Drinkliquor",
            "Atrialenlargement","Rightventricularhypertrophy" ,
            "Rheumatic", "Aorticvalve","Mitralvalve" ,"Arrhythmia",
            "Dementia", "Parkinson",
            "Adultseizuredisorder","Neurological" , "Thyroid","Endocrine" , 
            "Renal"  , "Gynecologic", "Emphysema" , "Pneumonia" ,"Asthma","Pulmonary",
            "Gout" ,"Degenerative", "Rheumatoidarthritis","Musculoskeletal" ,
            "Gallbladder" , "Gerd" ,  "Liver"  , "Gidisease" ,"Hematologicdisorder"  ,
            "Bleedingdisorder" , "Eye"  , "Ent" , "Skin", "Depression"  ,"Anxiety" ,
            "Psychosis" ,"Prostate" ,"Infectious",  "Fever" , 
            "Chronicbronchitis" , "COPD" )
      for(i in c){
        train_meta[,i]<- as.factor(train_meta[,i])
        train_meta[,i]<- as.logical(train_meta[,i] == 1)}
      {
        label(train_meta$Ejectionfraction)    <- "Ejection fraction" 
        label(train_meta$CoQ10)    <- "CoQ 10" 
        label(train_meta$Omega3)    <- "Omega 3"
        label(train_meta$AngiotensinIIantagonists)    <- "Angiotensin II antagonists"
        label(train_meta$Betablocking)    <- "Beta blocking"
        label(train_meta$Coronaryheartdisease)    <- "Coronary heart disease" 
        label(train_meta$Myocardialinfarction)    <- "Myocardial infarction"
        label(train_meta$Atrialfibrillation)    <- "Atrial fibrillation" 
        label(train_meta$Bloodglucose)    <- "Blood glucose" 
        label(train_meta$LDLcholesterol)    <- "LDL cholesterol" 
        label(train_meta$Numberofcigarettessmoked)    <- "Number of cigarettes smoked"
        label(train_meta$Creatinineserum)    <- "Creatinine serum" 
        label(train_meta$Averagediastolicbloodpressure)    <- "Average diastolic blood pressure" 
        label(train_meta$Leftventricularhypertrophy)    <- "Left ventricular hypertrophy" 
        label(train_meta$Fastingbloodglucose)    <- "Fasting blood glucose" 
        label(train_meta$HDLcholesterol)    <- "HDL cholesterol" 
        label(train_meta$Averagesystolicbloodpressure)    <- "Average systolic blood pressure" 
        label(train_meta$Totalcholesterol)    <- "Total cholesterol" 
        label(train_meta$Ventricularrate)    <- "Ventricular rate" 
        label(train_meta$Treatedforhypertension)    <- "Treated for hypertension" 
        label(train_meta$Treatedforlipids)    <- "Treated for lipids" 
        label(train_meta$Drinkbeer)    <- "Drink beer" 
        label(train_meta$Drinkwine)    <- "Drink wine" 
        label(train_meta$Drinkliquor)    <- "Drink liquor" 
        label(train_meta$Albuminurine)    <- "Albumin urine" 
        label(train_meta$Creatinineurine)    <- "Creatinine urine" 
        label(train_meta$HemoglobinA1cwholeblood)    <- "Hemoglobin A1c whole blood" 
        label(train_meta$Atrialenlargement)    <- "Atrial enlargement" 
        label(train_meta$Rightventricularhypertrophy)    <- "Right ventricular hypertrophy" 
        label(train_meta$Aorticvalve)    <- "Aortic valve" 
        label(train_meta$Mitralvalve)    <- "Mitral valve" 
        label(train_meta$Adultseizuredisorder)    <- "Adultseizure disorder"
        label(train_meta$Hematologicdisorder)    <- "Hematologic disorder"
        label(train_meta$Bleedingdisorder)    <- "Bleeding disorder"
        label(train_meta$Chronicbronchitis)    <- "Chronic bronchitis"
        label(train_meta$Creactiveprotein)    <- "C reactive protein" 
        label(train_meta$CD8T)    <- "CD8+ T cell" 
        label(train_meta$CD4T)    <- "CD4+ T cell"
        label(train_meta$NK)    <- "Natural killer cell"
        label(train_meta$Bcell)    <- "B cell"
        label(train_meta$Mono)    <- "Monocyte cell"
        label(train_meta$Gran)    <- "Granulocytes cell" 
      }
      
      {
        units(train_meta$Age)      <- "years"
        units(train_meta$Sleep)      <- "hours"
        units(train_meta$Numberofcigarettessmoked)      <- "day"
        units(train_meta$CD8T)      <- "proportions"
        units(train_meta$CD4T)      <- "proportions"
        units(train_meta$NK)      <- "proportions"
        units(train_meta$Bcell)      <- "proportions"
        units(train_meta$Mono)      <- "proportions"
        units(train_meta$Gran)      <- "proportions"
        train_meta$Heartfailure <- factor(train_meta$Heartfailure, levels=c(0, 1, 2), labels=c("NoChf", "Chf", "P-value"))
      }
      rndr <- function(x, name, ...) {
        if (length(x) == 0) {
          y <- train_meta[[name]]
          s <- rep("", length(render.default(x=y, name=name, ...)))
          if (is.numeric(y)) {
            p <- wilcox.test(y ~ train_meta$Heartfailure)$p.value #行wilcoxon=Mann-Whitney U
          } else {#t.test
            p <- chisq.test(table(y, droplevels(train_meta$Heartfailure)))$p.value
          }
          s[2] <- sub("<", "&lt;", format.pval(p, digits=3, eps=0.001))
          s
        } else {
          render.default(x=x, name=name, ...)
        }
      }
      rndr.strat <- function(label, n, ...) {
        ifelse(n==0, label, render.strat.default(label, n, ...))
      }
      table1(~ Gender+Age+
               CD8T+CD4T+NK+Bcell+Mono+Gran+
               Smoking+Numberofcigarettessmoked+BMI+Hight+Waist+Weight+
               Fastingbloodglucose+Bloodglucose+LDLcholesterol+HDLcholesterol+Totalcholesterol+
               Triglycerides+Ventricularrate+Averagediastolicbloodpressure+Averagesystolicbloodpressure+
               Treatedforhypertension+Treatedforlipids+
               Ejectionfraction+Creactiveprotein+Creatinineserum+Creatinineurine+
               Albuminurine+HemoglobinA1cwholeblood+  
               Drinkbeer+Drinkwine+Drinkliquor+Sleep+
               Omega3+Statin+Thiazides+Diuretic+Potassium+Aldosterone+Amiodarone+Vasodilators+CoQ10+
               Betablocking+AngiotensinIIantagonists+ACEI+Warfarin+Clopidogrel+Aspirin+Folicacid+
               Coronaryheartdisease+Myocardialinfarction+Diabetes+Atrialfibrillation+
               Stroke+Leftventricularhypertrophy+Atrialenlargement+Rightventricularhypertrophy+
               Rheumatic+Aorticvalve+Mitralvalve+Arrhythmia+Dementia+Parkinson+Adultseizuredisorder+
               Neurological+Thyroid+Endocrine+Renal+Gynecologic+Emphysema+Pneumonia+Asthma+Pulmonary+Gout+
               Degenerative+Rheumatoidarthritis+Musculoskeletal+Gallbladder+Gerd+Liver+
               Gidisease+Hematologicdisorder+Bleedingdisorder+Eye+Ent+Skin+Depression+
               Anxiety+Psychosis+Prostate+Infectious+Fever+Chronicbronchitis+COPD | Heartfailure,data=train_meta, 
             droplevels=F, render=rndr, render.strat=rndr.strat, overall=F)
    }
  }
  #rm missing > 20% or P>0.05
  {
    for(name in colnames(train_meta[,-c(19)])){
      y <- train_meta[[name]]
      if (is.numeric(y)) {
        p <- wilcox.test(y ~ train_meta$Heartfailure)$p.value #wilcoxon = Mann-Whitney U
      } else {#t.test
        p <- chisq.test(table(y, droplevels(train_meta$Heartfailure)))$p.value
      }
      if(p>0.05){
        print(name)
      }
    }
    #p value
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Ejectionfraction", "Omega3", "Statin", 
                                                      "Thiazides", "Potassium",
                             "Aldosterone", "Amiodarone", "Vasodilators", "CoQ10",
                             "Warfarin", "Clopidogrel", "Folicacid", "Myocardialinfarction", "Stroke",
                             "Numberofcigarettessmoked", "Smoking", "Leftventricularhypertrophy", "Hight", "Triglycerides", "Ventricularrate",
                             "Drinkbeer", "Drinkwine", "Drinkliquor", "Sleep", "Creatinineurine",
                             "Mitralvalve", "Arrhythmia", "Dementia",
                             "Parkinson", "Adultseizuredisorder", "Neurological", 
                             "Thyroid", "Endocrine", "Renal", "Gynecologic", "Emphysema",
                             "Pneumonia", "Asthma", "Pulmonary", "Gout", "Degenerative",
                             "Musculoskeletal", "Gallbladder", "Gerd", "Liver",
                             "Gidisease", "Hematologicdisorder", "Bleedingdisorder",
                             "Eye", "Ent", "Skin", "Depression", "Anxiety", "Psychosis",
                             "Prostate", "Infectious", "Fever", "Chronicbronchitis",
                             "COPD","CD8T","CD4T","NK","Bcell","Mono","Gran")]
    #missing
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Diabetes")]
    save(train_meta,file="train_meta.Rdata")
    # all 0
    train_meta <- train_meta[,!colnames(train_meta)  %in% c("Rightventricularhypertrophy")]
    save(train_meta,file="train_meta.Rdata")
  }
  #impute
  {
    library(tibble)
    library(impute)
    
    load(file="train_meta_raw.Rdata")
    load(file="train_meta.Rdata")
    train_meta <- train_meta_raw[,colnames(train_meta_raw) %in% c(colnames(train_meta))]
    #same feature is many NA (NA>1%)
    Patient_impute <- impute.knn(as.matrix(data.frame(t(train_meta))))
    Patient_impute <- data.frame(t(Patient_impute$data))
    #change 0,1,
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement")){
      Patient_impute[,i] <- round(Patient_impute[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein")){
      Patient_impute[,i] <- round(Patient_impute[,i],2)
    }
    write.csv(Patient_impute,"Patient_impute.csv")
  }
  #clinic correction
  {
    suppressMessages(silent <- lapply(c("readxl", "dplyr",
                                        "Hmisc","corrplot",
                                        "RColorBrewer","kableExtra"), library, character.only=T))
    
    Patient_impute <- read.table("Patient_impute.csv",sep=",",header = T,row.names = 1)
    colnames(Patient_impute) = c("Diuretic","Beta blocking",
                                 "Angiotensin II antagonists","ACEI",
                                 "Aspirin","Coronary heart disease","Heart failure",
                                 "Atrial fibrillation","Gender","Age","Blood glucose","BMI",
                                 "LDL cholesterol","Creatinine serum",
                                 "Average diastolic blood pressure","Fasting blood glucose",
                                 "HDL cholesterol","Average systolic blood pressure",
                                 "Total cholesterol","Waist","Weight",
                                 "Treated for hypertension","Treated for lipids",
                                 "Albumin urine","Hemoglobin a1c whole blood",
                                 "Atrial enlargement","Rheumatic","Aortic valve",
                                 "Rheumatoid arthritis","C reactive protein")
    cor<-data.matrix(Patient_impute)
    cor.m<-rcorr(cor)
    cor.r<-cor(cor)
    paletteLength <- 100
    myColor <- colorRampPalette(c("dodgerblue4", "white", "brown4"))(paletteLength)
    p.mat<-cor.mtest(cor.r)$p
    corrplot(cor.r, 
             order="AOE", 
             type="lower", 
             method = "circle", 
             hclust.method = "ward.D2", 
             outline=TRUE, 
             addrect = -1, 
             col = myColor, 
             tl.cex=0.5, 
             tl.col="black",  
             addgrid.col = NA)
    
  }
  #rm cor<0.8 feature
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
    Patient_impute <- Patient_impute[,!colnames(Patient_impute) %in% c("Blood glucose","LDL cholesterol","Waist","Weight")]#删除彼此相关性高，和chf相关性低的特征,还剩下41-5=36
    write.csv(Patient_impute,"Patient_impute_25.csv")
  }
  #norm
  {
    #normalization-not 0,1 feature
    normalization<-function(x){
      return((x-min(x))/(max(x)-min(x)))}
    #c= c("AGE8","BMI8","CREAT8","DBP8","FASTING_BG8","HDL8","SBP8","TC8,Albumin_urine","Hemoglobin_A1c_wholeblood","crp")
    for(i in c(10:17,20,21,26)){ 
      #print(colnames(Patient_impute[,i]))
      Patient_impute[,i] <- normalization(Patient_impute[,i])
    }
    write.csv(Patient_impute,"Patient_impute_25_nor.csv")
    
  }
}

#merge cpg and valid EHR
{
  load(file= "CorrectedBeta.Rdata")
  new_beta = data.frame(t(CorrectedBeta))
  new_beta[1:4,1:10]
  new_beta = rownames_to_column(new_beta,"Sample_Name")
  data_beta = read.table("sigCpGs.txt")#318
  #data_beta = read.table("new_champ/sigCpGs.txt")#32
  beta = new_beta[,colnames(new_beta) %in% data_beta$probe.id]
  Patient_impute <- read.csv("Patient_impute_25_nor.csv",head=T,row.names = 1)
  
  #beta+ehr
  train_data <- cbind(Patient_impute,new_beta)
  save(train_data,file="train_data.Rdata")
  #some beta+ehr
  train_data_sigCpGs <- cbind(Patient_impute,beta)
  save(train_data_sigCpGs,file="train_data_sigCpGs.Rdata")
}
#merge overlap cpg and valid EHR
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/new_champ/")
  #beta
  load(file= "CorrectedBeta.Rdata")
  new_beta = data.frame(t(CorrectedBeta))
  new_beta[1:4,1:10]
  load("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\diagnosis_chf/0622_nobootstrapping/champ/ID_overlap.Rdata")
  beta = new_beta[,colnames(new_beta) %in% dmp_over1]
  beta = new_beta[,colnames(new_beta) %in% dmp_over2]
  beta = new_beta[,colnames(new_beta) %in% dmp_over3]
  beta = new_beta[,colnames(new_beta) %in% dmp_over4]
  beta = new_beta[,colnames(new_beta) %in% dmp_over5]
  beta = new_beta[,colnames(new_beta) %in% dmp_over6]
  beta = new_beta[,colnames(new_beta) %in% dmp_over7]
  
  #valid EHR
  Patient_impute <- read.csv("../Patient_impute_25_nor.csv",head=T,row.names = 1)
  
  train_data <- cbind(Patient_impute,beta)
  save(train_data,file="train_data_1.Rdata")
  save(train_data,file="train_data_2.Rdata")
  save(train_data,file="train_data_3.Rdata")
  save(train_data,file="train_data_4.Rdata")
  save(train_data,file="train_data_5.Rdata")
  save(train_data,file="train_data_6.Rdata")
  save(train_data,file="train_data_7.Rdata")
}
#merge overlap pair cpg and valid EHR
{
  setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/new_champ/")
  #beta
  load(file= "CorrectedBeta.Rdata")
  new_beta = data.frame(t(CorrectedBeta))
  new_beta[1:4,1:10]
  load("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/20210628_pair/champ/ID_pair.Rdata")
  #sigDMV,sigCpGs,Bumphunter_dmp_early
  beta_pair1 = new_beta[,colnames(new_beta) %in% sigDMV$probe.id]
  beta_pair2 = new_beta[,colnames(new_beta) %in% sigCpGs$probe.id]
  beta_pair3 = new_beta[,colnames(new_beta) %in% Bumphunter_dmp_early$probe.id]
  
  #valid EHR
  Patient_impute <- read.csv("../Patient_impute_25_nor.csv",head=T,row.names = 1)
  
  train_data <- cbind(Patient_impute,beta_pair1)
  train_data <- cbind(Patient_impute,beta_pair2)
  train_data <- cbind(Patient_impute,beta_pair3)
  save(train_data,file="train_data_pair1.Rdata")
  save(train_data,file="train_data_pair2.Rdata")
  save(train_data,file="train_data_pair3.Rdata")
}
