from sklearn.cross_validation import KFold

kf = KFold(26, n_folds=3,
           shuffle=True, random_state=2016)
for train_index, test_index in kf:
    print train_index, test_index
