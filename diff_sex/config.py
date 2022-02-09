name = "dmp_lassoxgboost"
names= ""
SUB_DIR = "output/new_1126"

#ehr,cpg
#4years
#woman
#top
##50,25percentage,batch_size=100
##6,4 years ;batch_size=100

#top5,10,15,20,25
#TRAIN_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s.csv"%(name,names)
#TEST_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s_test.csv"%(name,names)

#old
#TRAIN_FILE = "data\\20201126deepfm_feature_%s%s.csv"%(name,names)
#TEST_FILE = "data\\20201126deepfm_feature_%s%s_test.csv"%(name,names)

#right sample 153
#TEST_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s_right_test.csv"%(name,names)

#Externaltest
#TEST_FILE = "data\\new_1126\\20210816deepfm_feature_%s%s_Externaltest.csv"%(name,names)#37
#TEST_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s_Externaltest2.csv"%(name,names)#36+689

#no_HFpEF
#TEST_FILE = "data\\new_1126\\20210816deepfm_feature_%s%s_no_HFpEFtest.csv"%(name,names)#37

#diff sex
TRAIN_FILE = "data\\new_1126\\20210817deepfm_feature_%s%s_sex.csv"%(name,names)
TEST_FILE = "data\\new_1126\\20210817deepfm_feature_%s%s_sex_test.csv"%(name,names)
sex = 0


NUM_SPLITS = 10
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
#CATEGORICAL_COLS = [
#"Diuretic"
#]
CATEGORICAL_COLS = [
"Sulfonamides","SEX"
]

#NUMERIC_COLS = [
#
##id3 = DMP lasso xgboost
#"Age","BMI","Creatinine.serum","Albumin.urine",
#"cg00522231","cg05845376","cg07041999","cg17766026","cg24205914",
#"cg08101977","cg25755428","cg05363438","cg13352914","cg03233656",
#"cg05481257","cg03556243","cg16781992","cg10083824","cg08614290",
#"cg21429551","cg00045910","cg10556349","cg21024264","cg27401945",
#"cg06344265","cg20051875","cg23299445","cg00495303","cg11853697"
#]
NUMERIC_COLS = [

#id3 = DMP lasso xgboost
"AGE8","BMI8","CREAT8","Albumin_urine",
"cg00522231","cg05845376","cg07041999","cg17766026","cg24205914",
"cg08101977","cg25755428","cg05363438","cg13352914","cg03233656",
"cg05481257","cg03556243","cg16781992","cg10083824","cg08614290",
"cg21429551","cg00045910","cg10556349","cg21024264","cg27401945",
"cg06344265","cg20051875","cg23299445","cg00495303","cg11853697"
]

IGNORE_COLS = [
    "ID", 
    "target"     
]
