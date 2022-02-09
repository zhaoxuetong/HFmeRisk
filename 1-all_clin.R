# 20210410
# Pre-processing of samples
# Get the sample from inside the phs000724, then get the sample with qualified quality and do the grouping of ejection fraction
# The latter will only do HFpEF analysis
# Then get the clinical features that we are concerned about more than 90
# Then get the time, get the exam8, eight years after the disease sample and control


library(rlang)
library(dplyr)
library("readxl")

#data
{
  setwd("H:/dbgap_CHD/ChildStudyConsentSet_phs000724.Framingham.v7.p10.c1.HMB-IRB-MDS/PhenotypeFiles")
  c1=read.table("phs000724.v7.pht004246.v2.p11.c1.Framingham_DNA_Methylation_Sample_Attributes_I.HMB-IRB-MDS.txt",sep="\t",header = T)
  setwd("H:/dbgap_CHD/ChildStudyConsentSet_phs000724.Framingham.v7.p10.c2.HMB-IRB-NPU-MDS/PhenotypeFiles")
  c2=read.table("phs000724.v7.pht004246.v2.p11.c2.Framingham_DNA_Methylation_Sample_Attributes_I.HMB-IRB-NPU-MDS.txt",sep="\t",header = T)
  data1 <- rbind(c1,c2)
  data1$SAMPID=strsplit(as.character(data1$SAMPID), "_724")
  data1 = data1[,c(2,3,8)]
  data1 <- filter(data1, PACKS_SET !="GEN3")
  data1 <- filter(data1, LABID !="")
  data1$SAMPID <- as.character(data1$SAMPID)#2725
  
  setwd("H:/dbgap_CHD/ChildStudyConsentSet_phs000724.Framingham.v7.p10.c1.HMB-IRB-MDS/PhenotypeFiles")
  c1=read.table("phs000724.v7.pht004247.v2.p11.c1.Framingham_DNA_Methylation_Sample_Attributes_II.HMB-IRB-MDS.txt",sep="\t",header = T)
  setwd("H:/dbgap_CHD/ChildStudyConsentSet_phs000724.Framingham.v7.p10.c2.HMB-IRB-NPU-MDS/PhenotypeFiles")
  c2=read.table("phs000724.v7.pht004247.v2.p11.c2.Framingham_DNA_Methylation_Sample_Attributes_II.HMB-IRB-NPU-MDS.txt",sep="\t",header = T)
  data2 <- rbind(c1,c2)
  data2$SAMPID=strsplit(as.character(data2$SAMPID), "_724")
  data2 = data2[,c(2,3,5,11)]
  data2$SAMPID = as.character(data2$SAMPID)#2782
  colnames(data2)[3] = "Sample_Well"
  data = merge(data1,data2,by="SAMPID",all.x=TRUE,all.y=TRUE)#2782;data2 have rep, data1 no rep
}
#Pre-processing data
{
  #qc-17
  data2_QC = filter(data2,QC_Comment != "")
  QC = unique(as.character(data2_QC$SAMPID))
  #rep-29
  index<-duplicated(data2$SAMPID)
  data2_index <-data2[index,]
  rep = unique(as.character(data2_index$SAMPID))
  data = filter(data, !SAMPID %in% c(rep,QC))#2626
  data$Sentrix_ID <- unlist(lapply(as.character(data$LABID.y),function(x){strsplit(x,'_')[[1]][[1]]}))
  #slide: data$Sentrix_ID=gsub("_[[:alnum:]]*", "", data$LABID.y)
  data$Sentrix_Position <- unlist(lapply(as.character(data$LABID.y),function(x){strsplit(x,'_')[[1]][[2]]}))
  #arrayPos:data$Sentrix_Position=gsub("[0-9]*_", "", data$LABID.y)
  data = data[,c(1,3,5,6,7,8)] #2626
}
save(data,file="data.Rdata")
#EF
{
  #c1
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  echo <- read_excel("phs000007.v30.pht002572.v6.p11.c1.t_echo_2008_m_0549s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  echo <- data.frame(echo)
  echo <- echo[,c(2,38,123)]
  #c2
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  echo_c2 <- read_excel("phs000007.v30.pht002572.v6.p11.c2.t_echo_2008_m_0549s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  echo_c2 <- data.frame(echo_c2)
  echo_c2 <- echo_c2[,c(2,38,123)]
  #rbind
  echo <- rbind(echo,echo_c2)
  colnames(echo)[2] <- c("EF")
  colnames(echo)[3] <- c("LVSF") 
  echo_1 <- filter(echo,EF<40)
  echo_2 <- filter(echo,EF>50)
  echo_3 <- filter(echo,EF<50 & EF>40)
}
HFpEF = merge(echo_2[,1:2],data,by.x="shareid",by.y="SAMPID")#20
HFrEF = merge(echo_1[,1:2],data,by.x="shareid",by.y="SAMPID")#2334
HFmrEF = merge(echo_3[,1:2],data,by.x="shareid",by.y="SAMPID")#38
save(HFpEF,file="HFpEF.Rdata")
#clinical
{
  #cvd
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  cvd_c1 <- read_excel("phs000007.v30.pht003316.v7.p11.c1.vr_survcvd_2014_a_1023s.HMB-IRB-MDS.xlsx",sheet=1, na = "NA", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  cvd_c2 <- read_excel("phs000007.v30.pht003316.v7.p11.c2.vr_survcvd_2014_a_1023s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "NA", skip = 10)
  cvd = rbind(cvd_c1,cvd_c2)
  cvd <- data.frame(cvd)
  cvd <- cvd[,-c(1,3)]
  head(cvd)
  #mi
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  MI_c1 <- read_excel("phs000007.v30.pht000309.v13.p11.c1.vr_soe_2016_a_1073s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  MI_c2 <- read_excel("phs000007.v30.pht000309.v13.p11.c2.vr_soe_2016_a_1073s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  MI = rbind(MI_c1,MI_c2)
  MI <- MI[,-c(1,4,5,6,8)]
  MIyes <-filter(MI , EVENT == "1" | EVENT == "2" | EVENT == "3" )
  MIyes <- MIyes[,-2]
  MIyes <- MIyes[!duplicated(MIyes$shareid), ] 
  MIyes$mi <- rep(1,nrow(MIyes))
  head(MIyes)
  colnames(MIyes)[2] ="midate"
  tmp1=merge(cvd,MIyes,by="shareid",all.x=TRUE,all.y=TRUE)
  tmp1$mi <- ifelse(is.na(tmp1$mi) == "FALSE" ,"1","0")
  #diba
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  DIAB_c1 <- read_excel("phs000007.v30.pht000041.v7.p11.c1.vr_diab_ex09_1_1002s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  DIAB_c1 <- data.frame(DIAB_c1)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  DIAB_c2 <- read_excel("phs000007.v30.pht000041.v7.p11.c2.vr_diab_ex09_1_1002s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  DIAB_c2 <- data.frame(DIAB_c2)
  DIAB = rbind(DIAB_c1,DIAB_c2)
  head(DIAB)
  DIAByes <- filter(DIAB,DIAB$CURR_DIAB1 == 1| DIAB$CURR_DIAB2 == 1|DIAB$CURR_DIAB3 == 1|DIAB$CURR_DIAB4 == 1|DIAB$CURR_DIAB5 == 1| DIAB$CURR_DIAB6 == 1| DIAB$CURR_DIAB7 == 1 | DIAB$CURR_DIAB8 == 1)
  DIAByes <- DIAByes[,c(2,11)]
  DIAByes[,2] <- "1"
  
  DIABno <- filter(DIAB,DIAB$CURR_DIAB1 != 1 & DIAB$CURR_DIAB2 != 1 & DIAB$CURR_DIAB3 != 1  & DIAB$CURR_DIAB4 != 1 & DIAB$CURR_DIAB5 != 1 &  DIAB$CURR_DIAB6 != 1 & DIAB$CURR_DIAB7 != 1 & DIAB$CURR_DIAB8 != 1)
  DIABno <- DIABno[,c(2,11)]
  DIABno[,2] <- "0"
  
  DIAB <- rbind(DIAByes,DIABno)
  DIAB <- DIAB[!duplicated(DIAB$shareid), ] 
  tmp2 = merge(tmp1,DIAB,by="shareid",all.x=TRUE,all.y=TRUE)
  #afx
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  afx_c1 <- read_excel("phs000007.v30.pht003315.v7.p11.c1.vr_survaf_2014_a_0987s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  afx_c2 <- read_excel("phs000007.v30.pht003315.v7.p11.c2.vr_survaf_2014_a_0987s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  afx = rbind(afx_c1,afx_c2)
  afx <- data.frame(afx)
  afx <- afx[,c(2,4,5)]
  head(afx)
  tmp3 = merge(tmp2,afx,by="shareid",all.x=TRUE,all.y=TRUE)
  #stroke
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  stroke_c1 <- read_excel("phs000007.v30.pht006023.v2.p11.c1.vr_survstk_2014_a_1031s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  stroke_c2 <- read_excel("phs000007.v30.pht006023.v2.p11.c2.vr_survstk_2014_a_1031s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  stroke = rbind(stroke_c1,stroke_c2)
  stroke <- data.frame(stroke)
  stroke <- stroke[,c(2,4,7)]
  head(stroke)
  tmp4 = merge(tmp3,stroke,by="shareid",all.x=TRUE,all.y=TRUE)
  #meta1
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta1_c1 <- read_excel("phs000007.v30.pht006027.v2.p11.c1.vr_wkthru_ex09_1_1001s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta1_c2 <- read_excel("phs000007.v30.pht006027.v2.p11.c2.vr_wkthru_ex09_1_1001s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  meta1 = rbind(meta1_c1,meta1_c2)
  c=c(1,206,207,3,20,29,38,47,56,62,71,80,89,96,105,114,128,137,146,155,161,170,188,197) #extract exam8 all information
  meta1 <- meta1[,c+1]
  head(meta1)
  meta1 <- data.frame(meta1)
  normalization<-function(x){
    return((x-min(x))/(max(x)-min(x)))}
  meta1$SEX <- normalization(meta1$SEX)
  tmp5 = merge(tmp4,meta1,by="shareid",all.x=TRUE,all.y=TRUE)
  #meta2
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta2_c1 <- read_excel("phs000007.v30.pht000747.v6.p11.c1.ex1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta2_c2 <- read_excel("phs000007.v30.pht000747.v6.p11.c2.ex1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  meta2 = rbind(meta2_c1,meta2_c2)
  c=c("shareid","H010","H071","H074","H077","H480")
  meta2 <- meta2[,colnames(meta2) %in% c]
  names <- c("shareid","aspirin","beer","wine","liquor","sleep")
  colnames(meta2) <- names
  tmp6 = merge(tmp5,meta2,by="shareid",all.x=TRUE,all.y=TRUE)
  #meta3
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta3_other2_c1 <- read_excel("phs000007.v30.pht000742.v6.p11.c1.fhslab1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta3_other2_c2 <- read_excel("phs000007.v30.pht000742.v6.p11.c2.fhslab1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  meta3_other2 = rbind(meta3_other2_c1,meta3_other2_c2)
  meta3_other2 <- meta3_other2[,c(2,10:12)]
  colnames(meta3_other2)[1] <- c("SAMPID") 
  colnames(meta3_other2)[2] <- c("Albumin_urine")
  colnames(meta3_other2)[3] <- c("Creatinine_urine")
  colnames(meta3_other2)[4] <- c("Hemoglobin_A1c_wholeblood")
  meta3_other2 <- data.frame(meta3_other2)
  head(meta3_other2)
  tmp7 = merge(tmp6,meta3_other2,by.x = "shareid",by.y="SAMPID",all.x=TRUE,all.y=TRUE)
  #meta4
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta3_other1_c1 <- read_excel("phs000007.v30.pht000747.v6.p11.c1.ex1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  meta3_other1_c2 <- read_excel("phs000007.v30.pht000747.v6.p11.c2.ex1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  meta3_other1 = rbind(meta3_other1_c1,meta3_other1_c2)
  meta3_other1 <- meta3_other1[,-1]
  meta3_other1 <- data.frame(meta3_other1)
  meta3_other1 <- meta3_other1[,c(1,251:292,450,451,453,456,459)]
  c <- c("ATRIAL_ENLARGEMENT","RVH","LVH","RHEUMATIC","AORTIC_VALVE","MITRAL_VALVE","OTHER_HEART",
         "ARRHYTHMIA","OTHER_PERIPHERAL_VASCULAR_DISEASE","OTHER_VASCULAR_DIAGNOSIS","DEMENTIA",
         "PARKINSON","ADULT_SEIZURE_DISORDER","NEUROLOGICAL","THYROID","ENDOCRINE","RENAL",
         "GYNECOLOGIC","EMPHYSEMA","PNEUMONIA","ASTHMA","PULMONARY","GOUT","DEGENERATIVE","RHEUMATOID_ARTHRITIS",
         "MUSCULOSKELETAL","GALLBLADDER","GERD","LIVER","GI_DISEASE","HEMATOLOGIC_DISORDER","BLEEDING_DISORDER","EYE",
         "ENT","SKIN","OTHER","DEPRESSION","ANXIETY","PSYCHOSIS","OTHER2","PROSTATE","INFECTIOUS",
         "FEVER","PNEUMONIA","CHRONIC_BRONCHITIS","EMPHYSEMA","COPD")
  colnames(meta3_other1)[2:48] <- tolower(c)
  meta3_other1[1:10,1:10]
  colnames(meta3_other1)[1] <- c("SAMPID") 
  #全部变为0,1的；去除2,3的
  for(i in 2:ncol(meta3_other1)){
    meta3_other1[,i] <- ifelse(meta3_other1[,i] == 0 ,0,1)
  }
  tmp8 = merge(tmp7,meta3_other1,by.x = "shareid",by.y="SAMPID",all.x=TRUE,all.y=TRUE)
  #crp
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  CRP_c1 <- read_excel("phs000007.v30.pht002888.v5.p11.c1.l_crp_2008_m_0477s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
  CRP_c1 = CRP_c1[,-5]
  setwd("E:/workplace/mywork/methy/dbgap/3_clin")
  CRP_c2 <- read_excel("phs000007.v30.pht002888.v5.p11.c2.l_crp_2008_m_0477s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
  CRP = rbind(CRP_c1,CRP_c2)
  CRP <- CRP[,c(2,4)]
  head(CRP)
  tmp9 = merge(tmp8,CRP,by="shareid",all.x=TRUE,all.y=TRUE)
  #medicine
  {
    #FOLIC
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      FOLIC_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      FOLIC_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      FOLIC =  rbind(FOLIC_c1,FOLIC_c2)
      FOLIC <- data.frame(FOLIC)
      FOLIC <- FOLIC[,c(2,10:17)]
      FOLIC <- filter(FOLIC, phrm_gp1 == "VITAMIN B12 AND FOLIC ACID" & chem_gp1 == "Folic acid and derivatives")
      
      data <- merge(FOLIC ,tmp9,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Folic_acid"
      colnames(data)[3] = "Folic_acid_amount"
      data$Folic_acid_amount <- ifelse(data$Folic_acid == 0 ,"0",data$Folic_acid_amount)
    }
    #aspirin-only cvd realtivel
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      aspirin_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      aspirin_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      aspirin = rbind(aspirin_c1,aspirin_c2)
      aspirin <- data.frame(aspirin)
      aspirin <- aspirin[,c(2,10:17)]
      aspirin <- filter(aspirin, system1 == "BLOOD AND BLOOD FORMING ORGANS" & 
                          ther_gp1 == "ANTITHROMBOTIC AGENTS" &  
                          chem_gp1 == "Platelet aggregation inhibitors excl. heparin" & chem_nm1 == "ACETYLSALICYLIC ACID")
      
      data <- merge(aspirin ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "aspirin"
      colnames(data)[3] = "aspirin_amount"
      data$aspirin_amount <- ifelse(data$aspirin == 0 ,"0",data$aspirin_amount)
    }
    #clopidogrel
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      clopidogrel_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      clopidogrel_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      clopidogrel = rbind(clopidogrel_c1,clopidogrel_c2)
      clopidogrel <- data.frame(clopidogrel)
      clopidogrel <- clopidogrel[,c(2,10:17)]
      clopidogrel <- filter(clopidogrel, system1 == "BLOOD AND BLOOD FORMING ORGANS" & 
                              ther_gp1 == "ANTITHROMBOTIC AGENTS" &  
                              chem_gp1 == "Platelet aggregation inhibitors excl. heparin" & chem_nm1 == "CLOPIDOGREL")
      
      data <- merge(clopidogrel ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "clopidogrel"
      colnames(data)[3] = "clopidogrel_amount"
      data$clopidogrel_amount <- ifelse(data$clopidogrel == 0 ,"0",data$clopidogrel_amount)
    }
    #warfarin
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      warfarin_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      warfarin_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      warfarin = rbind(warfarin_c1,warfarin_c2)
      warfarin <- data.frame(warfarin)
      warfarin <- warfarin[,c(2,10:17)]
      warfarin <- filter(warfarin, system1 == "BLOOD AND BLOOD FORMING ORGANS" & 
                           ther_gp1 == "ANTITHROMBOTIC AGENTS" &  
                           chem_gp1 == "Vitamin K antagonists" )
      
      data <- merge(warfarin ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "warfarin"
      colnames(data)[3] = "warfarin_amount"
      data$warfarin_amount <- ifelse(data$warfarin == 0 ,"0",data$warfarin_amount)
    }
    #ACEI
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      ACEI_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      ACEI_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      ACEI = rbind(ACEI_c1,ACEI_c2)
      ACEI <- data.frame(ACEI)
      ACEI <- ACEI[,c(2,10:17)]
      ACEI <- filter(ACEI, system1 == "CARDIOVASCULAR SYSTEM" & 
                       ther_gp1 == "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM" &  
                       phrm_gp1 == "ACE INHIBITORS, PLAIN" )
      
      data <- merge(ACEI ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "ACEI"
      colnames(data)[3] = "ACEI_amount"
      data$ACEI_amount <- ifelse(data$ACEI == 0 ,"0",data$ACEI_amount)
    }
    #Angiotensin_II_antagonists
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Angiotensin_II_antagonists_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Angiotensin_II_antagonists_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      Angiotensin_II_antagonists = rbind(Angiotensin_II_antagonists_c1,Angiotensin_II_antagonists_c2)
      Angiotensin_II_antagonists <- data.frame(Angiotensin_II_antagonists)
      Angiotensin_II_antagonists <- Angiotensin_II_antagonists[,c(2,10:17)]
      Angiotensin_II_antagonists <- filter(Angiotensin_II_antagonists, system1 == "CARDIOVASCULAR SYSTEM" & 
                                             ther_gp1 == "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM" &  
                                             phrm_gp1 == "ANGIOTENSIN II ANTAGONISTS, PLAIN" )
      
      data <- merge(Angiotensin_II_antagonists ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Angiotensin_II_antagonists"
      colnames(data)[3] = "Angiotensin_II_antagonists_amount"
      data$Angiotensin_II_antagonists_amount <- ifelse(data$Angiotensin_II_antagonists == 0 ,"0",data$Angiotensin_II_antagonists_amount)
    }
    #Beta_blocking
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Beta_blocking_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Beta_blocking_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      Beta_blocking = rbind(Beta_blocking_c1,Beta_blocking_c2)
      Beta_blocking <- data.frame(Beta_blocking)
      Beta_blocking <- Beta_blocking[,c(2,10:17)]
      Beta_blocking <- filter(Beta_blocking, system1 == "CARDIOVASCULAR SYSTEM" & 
                                ther_gp1 == "BETA BLOCKING AGENTS" )
      
      data <- merge(Beta_blocking ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Beta_blocking"
      colnames(data)[3] = "Beta_blocking_amount"
      data$Beta_blocking_amount <- ifelse(data$Beta_blocking == 0 ,"0",data$Beta_blocking_amount)
      
    }
    #CO_Q_10
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      CO_Q_10_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      CO_Q_10_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      CO_Q_10 = rbind(CO_Q_10_c1,CO_Q_10_c2)
      CO_Q_10 <- data.frame(CO_Q_10)
      CO_Q_10 <- CO_Q_10[,c(2,10:17)]
      CO_Q_10 <- filter(CO_Q_10, system1 == "CARDIOVASCULAR SYSTEM" & 
                          ther_gp1 == "CARDIAC THERAPY" &
                          phrm_gp1 == "OTHER CARDIAC PREPARATIONS")
      
      data <- merge(CO_Q_10 ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "CO_Q_10"
      colnames(data)[3] = "CO_Q_10_amount"
      data$CO_Q_10_amount <- ifelse(data$CO_Q_10 == 0 ,"0",data$CO_Q_10_amount)
    }
    #vasodilators
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      vasodilators_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      vasodilators_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      vasodilators = rbind(vasodilators_c1,vasodilators_c2)
      vasodilators <- data.frame(vasodilators)
      vasodilators <- vasodilators[,c(2,10:17)]
      vasodilators <- filter(vasodilators, system1 == "CARDIOVASCULAR SYSTEM" & 
                               ther_gp1 == "CARDIAC THERAPY" &
                               phrm_gp1 == "VASODILATORS USED IN CARDIAC DISEASES")
      
      data <- merge(vasodilators ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "vasodilators"
      colnames(data)[3] = "vasodilators_amount"
      data$vasodilators_amount <- ifelse(data$vasodilators == 0 ,"0",data$vasodilators_amount)
    }
    #amiodarone
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      amiodarone_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      amiodarone_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      amiodarone = rbind(amiodarone_c1,amiodarone_c2)
      amiodarone <- data.frame(amiodarone)
      amiodarone <- amiodarone[,c(2,10:17)]
      amiodarone <- filter(amiodarone, system1 == "CARDIOVASCULAR SYSTEM" & 
                             ther_gp1 == "CARDIAC THERAPY" &
                             phrm_gp1 == "ANTIARRHYTHMICS, CLASS I AND III" &
                             chem_nm1 == "AMIODARONE")
      
      data <- merge(amiodarone ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "amiodarone"
      colnames(data)[3] = "amiodarone_amount"
      data$amiodarone_amount <- ifelse(data$amiodarone == 0 ,"0",data$amiodarone_amount)
    }
    #Aldosterone antagonists
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Aldosterone_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Aldosterone_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      Aldosterone = rbind(Aldosterone_c1,Aldosterone_c2)
      Aldosterone <- data.frame(Aldosterone)
      Aldosterone <- Aldosterone[,c(2,10:17)]
      Aldosterone <- filter(Aldosterone, system1 == "CARDIOVASCULAR SYSTEM" & 
                              ther_gp1 == "DIURETICS" &
                              chem_gp1 == "Aldosterone antagonists")
      
      data <- merge(vasodilators ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Aldosterone"
      colnames(data)[3] = "Aldosterone_amount"
      data$Aldosterone_amount <- ifelse(data$Aldosterone == 0 ,"0",data$Aldosterone_amount)
      
    }
    #potassium_sparing_diuretic
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      potassium_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      potassium_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      potassium = rbind(potassium_c1,potassium_c2)
      potassium <- data.frame(potassium)
      potassium <- potassium[,c(2,10:17)]
      potassium <- filter(potassium, system1 == "CARDIOVASCULAR SYSTEM" & 
                            ther_gp1 == "DIURETICS" &
                            chem_gp1 == "Other potassium-sparing agents")
      
      data <- merge(potassium ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "potassium"
      colnames(data)[3] = "potassium_amount"
      data$potassium_amount <- ifelse(data$potassium == 0 ,"0",data$potassium_amount)
    }
    #Sulfonamides
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Sulfonamides_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Sulfonamides_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      Sulfonamides = rbind(Sulfonamides_c1,Sulfonamides_c2)
      Sulfonamides <- data.frame(Sulfonamides)
      Sulfonamides <- Sulfonamides[,c(2,10:17)]
      Sulfonamides <- filter(Sulfonamides, system1 == "CARDIOVASCULAR SYSTEM" & 
                               ther_gp1 == "DIURETICS" &
                               chem_gp1 == "Sulfonamides, plain")
      
      data <- merge(Sulfonamides ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Sulfonamides"
      colnames(data)[3] = "Sulfonamides_amount"
      data$Sulfonamides_amount <- ifelse(data$Sulfonamides == 0 ,"0",data$Sulfonamides_amount)
      
    }
    #Thiazides
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Thiazides_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      Thiazides_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      Thiazides = rbind(Thiazides_c1,Thiazides_c2)
      Thiazides <- data.frame(Thiazides)
      Thiazides <- Thiazides[,c(2,10:17)]
      Thiazides <- filter(Thiazides, system1 == "CARDIOVASCULAR SYSTEM" & 
                            ther_gp1 == "DIURETICS" &
                            chem_gp1 == "Thiazides, plain")
      
      data <- merge(Thiazides ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "Thiazides"
      colnames(data)[3] = "Thiazides_amount"
      data$Thiazides_amount <- ifelse(data$Thiazides == 0 ,"0",data$Thiazides_amount)
      
    }
    #statin
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      STATIN_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      STATIN_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      STATIN = rbind(STATIN_c1,STATIN_c2)
      STATIN <- data.frame(STATIN)
      STATIN <- STATIN[,c(2,10:17)]
      STATIN1 <- filter(STATIN, system1 == "CARDIOVASCULAR SYSTEM" & 
                          ther_gp1 == "LIPID MODIFYING AGENTS" &
                          chem_gp1 == "HMG CoA reductase inhibitors")
      STATIN2 <- filter(STATIN, system1 == "CARDIOVASCULAR SYSTEM" & 
                          ther_gp1 == "LIPID MODIFYING AGENTS" &
                          chem_gp1 == "Other lipid modifying agents" & chem_nm1 == "EZETIMIBE" & MEDNAME == "SIMVASTATIN" |MEDNAME == "SIMVASTATIN)")
      STATIN <- rbind(STATIN1,STATIN2)
      data <- merge(STATIN ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "STATIN"
      colnames(data)[3] = "STATIN_amount"
      data$STATIN_amount <- ifelse(data$STATIN == 0 ,"0",data$STATIN_amount)
    }
    #OMEGA_3
    {
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      OMEGA_3_c1 <- read_excel("phs000007.v30.pht000828.v6.p11.c1.meds1_8s.HMB-IRB-MDS.xlsx",sheet=1, na = "", skip = 10)
      setwd("E:/workplace/mywork/methy/dbgap/3_clin")
      OMEGA_3_c2 <- read_excel("phs000007.v30.pht000828.v6.p11.c2.meds1_8s.HMB-IRB-NPU-MDS.xlsx",sheet=1, na = "", skip = 10)
      OMEGA_3 = rbind(OMEGA_3_c1,OMEGA_3_c2)
      OMEGA_3 <- data.frame(OMEGA_3)
      OMEGA_3 <- OMEGA_3[,c(2,10:17)]
      OMEGA_3 <- filter(OMEGA_3, system1 == "CARDIOVASCULAR SYSTEM" & 
                          ther_gp1 == "LIPID MODIFYING AGENTS" &
                          chem_gp1 == "Other lipid modifying agents" & chem_nm1 == "OMEGA-3-TRIGLYCERIDES")
      data <- merge(OMEGA_3 ,data,by="shareid",all.y=TRUE)
      data$chem_gp1 <- ifelse(is.na(data$IDTYPE) == "FALSE" ,"1","0")
      data <- data[,-c(2:4,6:8)]
      colnames(data)[2] = "OMEGA_3"
      colnames(data)[3] = "OMEGA_3_amount"
      data$OMEGA_3_amount <- ifelse(data$OMEGA_3 == 0 ,"0",data$OMEGA_3_amount)
    }
    dim(data)
    #15285,125
  }
  #duplicated--no among
  data <- data[!duplicated(data[,-c(3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33)]),]#15154   125
  save(data,file="E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/all_clin.Rdata")
  data_exam8 = filter(data,DATE8 != "NA")#3000  125
  data_exam8$mi = ifelse(data_exam8$mi == 1 & data_exam8$midate<data_exam8$DATE8,1,0)
  data_exam8$cvd = ifelse(data_exam8$cvd == 1 & data_exam8$cvddate<data_exam8$DATE8,1,0)
  data_exam8$chd = ifelse(data_exam8$chd == 1 & data_exam8$chddate<data_exam8$DATE8,1,0)
  data_exam8$afx = ifelse(data_exam8$afx == 1 & data_exam8$afxdate<data_exam8$DATE8,1,0)
  data_exam8$stroke = ifelse(data_exam8$stroke == 1 & data_exam8$strokedate<data_exam8$DATE8,1,0)
  save(data_exam8,file="E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary/data_exam8.Rdata")
}
setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
load(file="data_exam8.Rdata")
HFpEF_exam8 = merge(HFpEF,data_exam8,by="shareid")#2334  131
save(HFpEF_exam8,file="HFpEF_exam8.Rdata")
#time
{
  library(dplyr)
  library(tibble)
  EHR = HFpEF_exam8
  
  #clin_data_control <- filter(EHR, chf == 0 ,(chfdate - DATE8 ) >8*365) 
  clin_data_control <- filter(EHR, (chfdate - DATE8 ) >8*365) 
  summary((clin_data_control$chfdate - clin_data_control$DATE8))
  out1 = filter(EHR, chf == 1 & chfdate < DATE8 )
  #clin_data_chf <- filter(EHR, chf == 1 & (chfdate >= DATE8))
  clin_data_chf <- filter(EHR, chf == 1 & (chfdate >= DATE8) & (chfdate - DATE8 ) <8*365)
  summary((clin_data_chf$chfdate - clin_data_chf$DATE8))
  out2 = filter(EHR, chf == 0 & (chfdate >= DATE8) & (chfdate - DATE8 ) <=8*365 )
  # tmp = filter(EHR, chf == 1 & (chfdate - DATE8 <= 5*365) & (chfdate >= DATE8))
  # summary(tmp$chfdate - tmp$DATE8)
  clin_data_chf_control <- rbind(clin_data_chf,clin_data_control)#989
  #20211220 cosoring
  clin_data_chf_control <- filter(EHR,!EHR$shareid %in% out1$shareid)
}
table(clin_data_chf_control$PACKS_SET)
#GEN3  JHU  UMN 
#0  171  797 
setwd("E:\\workplace\\mywork\\methy\\dbgap\\chf\\data_chf_contr\\early_chf\\c1_UMN_JHU\\train_UMN_tset_JHU/1123_dataSummary")
save(clin_data_chf_control,file="clin_data_chf_control.Rdata")