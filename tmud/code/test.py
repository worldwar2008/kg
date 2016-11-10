import pandas as pd
import numpy as np


print "Reading label_categories"
labelc_df = pd.read_csv("../input/label_categories.csv")
print labelc_df.head(3)
print "Reading app_labels"

appl_df = pd.read_csv("../input/app_labels.csv")
print appl_df.head(3)
print "Unique label type"
unique_label_id = np.unique(appl_df['label_id'])
print len(np.unique(appl_df['label_id']))
for label_id in unique_label_id:
    print labelc_df[labelc_df["label_id"]==label_id].as_matrix()[0]


