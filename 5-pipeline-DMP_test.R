#merge EHR and CpG
{
  library(impute)
  library(tibble)
  library(dplyr)
  library(impute)
  #JHU
  {
    setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
    #cell 
    load(file="test/CellFraction.Rdata")
    load(file="HFpEF.Rdata")
    data_cpg_test = filter(HFpEF,PACKS_SET == "JHU")
    data_cpg_chf <- filter(data_cpg_test , chf == 1)
    data_cpg_nochf <- filter(data_cpg_test , chf == 0)
    data_cpg <- rbind(data_cpg_chf,data_cpg_nochf)
    data_cpg$Sample_Name <- paste(rep(c("chf","control"),c(nrow(data_cpg_chf),nrow(data_cpg_nochf))),
                                  c(c(1:nrow(data_cpg_chf)),c(1:nrow(data_cpg_nochf))),sep = "")
    
    test_meta  <- data_cpg[,-c(4:7)]
    test_meta <- test_meta[,c(128,1:127)]
    test_meta_raw <- cbind(test_meta,CellFraction)
    {
      colnames(test_meta_raw)= c("Sample_Name","shareid",
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
    setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/test/")
    save(test_meta_raw,file="test_meta_raw.Rdata")
    test_meta = test_meta_raw[,!colnames(test_meta_raw) %in%  c("PACKS_SET","Omega3amount","Statinamount","Thiazidesamount","Diureticamount",
                                                                "Potassiumamount" , "Aldosteroneamount" , "Amiodaroneamount",
                                                                "Vasodilatorsamount","CoQ10amount","Betablockingamount", 
                                                                "AngiotensinIIantagonistsamount", "ACEIamount" , "Warfarinamount" , 
                                                                "Clopidogrelamount" , "Aspirinamount" , "Folicacidamount" ,"chddate" ,
                                                                "chfdate" ,"cvddate" ,"midate" ,"afxdate" ,"strokedate" ,"DATE8","lvh","cvd",
                                                                "DATE9","aspirin.1","other_heart","other_peripheral_vascular_disease",
                                                                "other_vascular_diagnosis","other","other2","pneumonia.1","emphysema.1")]
  }
}

#EHR
{
  #sex
  test_meta$Gender <- ifelse(test_meta$Gender == 1,"F","M")
  test_meta = test_meta[,-c(1,2)]
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
        test_meta[,i]<- as.factor(test_meta[,i])
        test_meta[,i]<- as.logical(test_meta[,i] == 1)}
      {
        label(test_meta$Ejectionfraction)    <- "Ejection fraction" 
        label(test_meta$CoQ10)    <- "CoQ 10" 
        label(test_meta$Omega3)    <- "Omega 3"
        label(test_meta$AngiotensinIIantagonists)    <- "Angiotensin II antagonists"
        label(test_meta$Betablocking)    <- "Beta blocking"
        label(test_meta$Coronaryheartdisease)    <- "Coronary heart disease" 
        label(test_meta$Myocardialinfarction)    <- "Myocardial infarction"
        label(test_meta$Atrialfibrillation)    <- "Atrial fibrillation" 
        label(test_meta$Bloodglucose)    <- "Blood glucose" 
        label(test_meta$LDLcholesterol)    <- "LDL cholesterol" 
        label(test_meta$Numberofcigarettessmoked)    <- "Number of cigarettes smoked"
        label(test_meta$Creatinineserum)    <- "Creatinine serum" 
        label(test_meta$Averagediastolicbloodpressure)    <- "Average diastolic blood pressure" 
        label(test_meta$Leftventricularhypertrophy)    <- "Left ventricular hypertrophy" 
        label(test_meta$Fastingbloodglucose)    <- "Fasting blood glucose" 
        label(test_meta$HDLcholesterol)    <- "HDL cholesterol" 
        label(test_meta$Averagesystolicbloodpressure)    <- "Average systolic blood pressure" 
        label(test_meta$Totalcholesterol)    <- "Total cholesterol" 
        label(test_meta$Ventricularrate)    <- "Ventricular rate" 
        label(test_meta$Treatedforhypertension)    <- "Treated for hypertension" 
        label(test_meta$Treatedforlipids)    <- "Treated for lipids" 
        label(test_meta$Drinkbeer)    <- "Drink beer" 
        label(test_meta$Drinkwine)    <- "Drink wine" 
        label(test_meta$Drinkliquor)    <- "Drink liquor" 
        label(test_meta$Albuminurine)    <- "Albumin urine" 
        label(test_meta$Creatinineurine)    <- "Creatinine urine" 
        label(test_meta$HemoglobinA1cwholeblood)    <- "Hemoglobin A1c whole blood" 
        label(test_meta$Atrialenlargement)    <- "Atrial enlargement" 
        label(test_meta$Rightventricularhypertrophy)    <- "Right ventricular hypertrophy" 
        label(test_meta$Aorticvalve)    <- "Aortic valve" 
        label(test_meta$Mitralvalve)    <- "Mitral valve" 
        label(test_meta$Adultseizuredisorder)    <- "Adultseizure disorder"
        label(test_meta$Hematologicdisorder)    <- "Hematologic disorder"
        label(test_meta$Bleedingdisorder)    <- "Bleeding disorder"
        label(test_meta$Chronicbronchitis)    <- "Chronic bronchitis"
        label(test_meta$Creactiveprotein)    <- "C reactive protein" 
        label(test_meta$CD8T)    <- "CD8+ T cell" 
        label(test_meta$CD4T)    <- "CD4+ T cell"
        label(test_meta$NK)    <- "Natural killer cell"
        label(test_meta$Bcell)    <- "B cell"
        label(test_meta$Mono)    <- "Monocyte cell"
        label(test_meta$Gran)    <- "Granulocytes cell" 
      }
      
      {
        units(test_meta$Age)      <- "years"
        units(test_meta$Sleep)      <- "hours"
        units(test_meta$Numberofcigarettessmoked)      <- "day"
        units(test_meta$CD8T)      <- "proportions"
        units(test_meta$CD4T)      <- "proportions"
        units(test_meta$NK)      <- "proportions"
        units(test_meta$Bcell)      <- "proportions"
        units(test_meta$Mono)      <- "proportions"
        units(test_meta$Gran)      <- "proportions"
        test_meta$Heartfailure <- factor(test_meta$Heartfailure, levels=c(0, 1, 2), labels=c("NoChf", "Chf", "P-value"))
      }
      rndr <- function(x, name, ...) {
        if (length(x) == 0) {
          y <- test_meta[[name]]
          s <- rep("", length(render.default(x=y, name=name, ...)))
          if (is.numeric(y)) {
            p <- wilcox.test(y ~ test_meta$Heartfailure)$p.value #wilcoxon = Mann-Whitney U
          } else {#t.test
            p <- chisq.test(table(y, droplevels(test_meta$Heartfailure)))$p.value
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
               Anxiety+Psychosis+Prostate+Infectious+Fever+Chronicbronchitis+COPD | Heartfailure,data=test_meta, 
             droplevels=F, render=rndr, render.strat=rndr.strat, overall=F)
    }
    
  }
  #rm- same of UMN
  {
    #p value
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Ejectionfraction", "Omega3", "Statin", 
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
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Diabetes")]
    # all 0
    test_meta <- test_meta[,!colnames(test_meta)  %in% c("Rightventricularhypertrophy")]
    save(test_meta,file="test_meta.Rdata")
  }
  #impute
  {
    library(tibble)
    library(impute)
    
    load(file="test_meta_raw.Rdata")
    load(file="test_meta.Rdata")
    test_meta <- test_meta_raw[,colnames(test_meta_raw) %in% c(colnames(test_meta))]
    #same feature is many NA (NA>1%)
    Patient_impute_test <- impute.knn(as.matrix(data.frame(t(test_meta))))
    Patient_impute_test <- data.frame(t(Patient_impute_test$data))
    #change 0,1,
    for(i in c("LDLcholesterol","Fastingbloodglucose","Atrialenlargement")){
      Patient_impute_test[,i] <- round(Patient_impute_test[,i],0)
    }
    for(i in c("BMI","Waist","Albuminurine","Creactiveprotein")){
      Patient_impute_test[,i] <- round(Patient_impute_test[,i],2)
    }
    write.csv(Patient_impute_test,"Patient_impute_test.csv")
  }
  Patient_impute_test <- read.table("Patient_impute_test.csv",sep=",",header = T,row.names = 1)
  colnames(Patient_impute_test)= c("Diuretic","Beta blocking",
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
  ##rm cor<0.8 feature same UMN
  Patient_impute_test <- Patient_impute_test[,!colnames(Patient_impute_test) %in% c("Blood glucose","LDL cholesterol","Waist","Weight")]#删除彼此相关性高，和chf相关性低的特征,还剩下41-5=36
  write.csv(Patient_impute_test,"Patient_impute_test_25.csv")
  #norm
  {
    #normalization-not 0,1 feature
    normalization<-function(x){
      return((x-min(x))/(max(x)-min(x)))}
    #c= c("AGE8","BMI8","CREAT8","DBP8","FASTING_BG8","HDL8","SBP8","TC8,Albumin_urine","Hemoglobin_A1c_wholeblood","crp")
    for(i in c(10:17,20,21,26)){ 
      Patient_impute_test[,i] <- normalization(Patient_impute_test[,i])
    }
    write.csv(Patient_impute_test,"Patient_impute_test_25_nor.csv")
    
  }
}

#merge cpg and valid EHR
{
  #beta
  load(file= "CorrectedBeta.Rdata")
  new_beta_test = data.frame(t(CorrectedBeta))
  new_beta_test[1:4,1:10]
  
  #valid EHR
  Patient_impute <- read.csv("Patient_impute_test_25_nor.csv",head=T,row.names = 1)
  #beta+ehr
  test_data <- cbind(Patient_impute,new_beta_test)
  save(test_data,file="test_data.Rdata")
}