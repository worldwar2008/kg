import pandas as pd
import numpy as np

df = pd.read_csv("./driver_imgs_list.csv")
print df.head()

print np.unique(df['subject'])