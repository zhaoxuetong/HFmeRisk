name = ""#_mirna,
names= ""
#ehr,cpg
#4years
#woman
#top
##50,25percentage,batch_size=100
##6,4 years ;batch_size=100

#top5,10,15,20,25
#TRAIN_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s.csv"%(name,names)
#TEST_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s_test.csv"%(name,names)
SUB_DIR = "output/mirna"
TRAIN_FILE = "data\\mirna\\deepfm_feature_mirna_lassoxgboost%s%s.csv"%(name,names)
TEST_FILE = "data\\mirna\\deepfm_feature_mirna_lassoxgboost%s%s_test.csv"%(name,names)

#right sample 153
#TEST_FILE = "data\\new_1126\\20210707deepfm_feature_%s%s_right_test.csv"%(name,names)

NUM_SPLITS = 10
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
#CATEGORICAL_COLS = [
#"Diuretic"
#]
CATEGORICAL_COLS = [
#after limma
"Aortic.Valve","Sulfonamides","Beta.blocking"

#no limma
#"Beta_blocking","chd","Sulfonamides"

#no limma,sample
#"afx","Beta_blocking","atrial_enlargement","eye","asthma"
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
#after limma
"Creatinine.serum","Average.systolic.blood.pressure","miR_142_3p", "Albumin.urine",
"miR_886_3p","miR_125b_5p","miR_186_5p_a1","AGE","miR_218_5p","miR_339_5p",
"Hemoglobin.A1c.wholeblood","miR_296_5p","miR_145_5p","FASTING.blood.glucose","miR_128"

#no limma
#"AGE8","CREAT8","WGT8","miR_885_5p","miR_29c_5p","miR_223_3p","miR_339_5p",
#"miR_886_3p","miR_128","miR_145_5p","miR_296_5p","miR_1296"

#no limma,sample
#"AGE8","BMI8","miR_223_3p","CREAT8", "miR_128", "DBP8","miR_1296",
#"miR_210","miR_720","miR_339_5p","miR_29c_5p","miR_152","miR_624_5p"
]

IGNORE_COLS = [
    "ID", 
    "target"     
]
