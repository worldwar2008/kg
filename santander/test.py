# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing,ensemble
from sklearn.model_selection import cross_val_score

# columns to be used as features #
feature_cols = ["ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", "segmento", "canal_entrada","renta"]
dtoype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'floatOB16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'] 

def detectRenta(test_df):
    print test_df["renta"].dtype
    print np.unique(test_df["renta"])[:500]
    print "-"*10
    print "需要检测异常的数据，然后用合适的值替换掉这样的数据，把字符串转化成数字"
    print test_df[test_df["renta"]=="         NA"]

    new_test_df = test_df.replace("         NA",0)
    print new_test_df.head

def detect_Common(test_df,field):
    print "开始检测%s字段"%field

    print np.unique(test_df[field])[:500]



if __name__ == "__main__":
    data_path = "./input/"
    train_file = data_path + "train_ver2.csv"
    test_file = data_path + "test_ver2.csv"
    train_size = 13647309
    nrows = 10000 # change this value to read more rows from train

    start_index = train_size - nrows
    test_df = pd.read_csv(test_file)
    test_df =test_df.fillna(-99)
    print test_df.count()
    d = ["ind_empleado","pais_residencia","sexo",
         "age","antiguedad","nomprov",
         "segmento","canal_entrada"]
    for item in d:
        detect_Common(test_df,item)
        
    
