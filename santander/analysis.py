import numpy as np
import pandas as pd

limit_rows = 700000
df = pd.read_csv("./input/train_ver2.csv",dtype={"sexo":str,
                                          "ind_nuevo":str,
                                          "ult_fec_cli_lt":str,
                                          "indext":str},nrows=limit_rows)
unique_ids = pd.Series(df["ncodpers"].unique())
limit_people = 1e4
unique_id = unique_ids.sample(n=limit_people)
df = df[df.ncodpers.isin(unique_id)]
print df.describe()
