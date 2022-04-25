from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import os

train = pd.read_feather('/data/xxl/dataset/train.feather')

def make_dataset(train):
    combination_features = ['f_231-f_250', 'f_118-f_280', 'f_155-f_297','f_25-f_237',
                            'f_179-f_265', 'f_119-f_270', 'f_71-f_197', 'f_21-f_65']
    for f in combination_features:
        f1, f2 = f.split('-')
        train[f] = train[f1] + train[f2]
    drop_features = ['f_148', 'f_72', 'f_49', 'f_205', 'f_228', 'f_97', 'f_262', 'f_258']
    train = train.drop(drop_features, axis=1)
    return train

train = make_dataset(train)

f_col = train.drop(['investment_id','target','time_id'],axis=1).columns

target = 'target'
if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train","test"], p =[.8, .2], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
test_indices = train[train.Set=="test"].index  

unused_feat = ['Set']
features = [ col for col in train.columns if col not in unused_feat+[target]]

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices].reshape(-1, 1)

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices].reshape(-1, 1)

clf = TabNetRegressor(cat_emb_dim=1, cat_idxs= [i for i, f in enumerate(features) if f in ['investment_id','time_id']],optimizer_params = dict(lr=0.02),
n_a=16,n_d=16,gamma =1.4690246460970766,optimizer_fn = Adam,scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
        scheduler_fn = CosineAnnealingWarmRestarts)

from pytorch_tabnet.augmentations import RegressionSMOTE
aug = RegressionSMOTE(p=0.2)

class PearsonCorrelation(Metric):
    def __init__(self):
        self._name = 'pearson_corr'
        self._maximize = True

    def __call__(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        x_diff = x - np.mean(x)
        y_diff = y - np.mean(y)
        return np.dot(x_diff, y_diff)/(np.sqrt(sum(x_diff**2))*np.sqrt(sum(y_diff**2)))

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['rmse','pearson_corr'],
    max_epochs=10,
    patience=50,
    batch_size=1024*10, virtual_batch_size=256*5,
    num_workers=0,
    drop_last=False,
    augmentations=aug,
) 

preds = clf.predict(X_test)

y_true = y_test

test_score = mean_squared_error(y_pred=preds, y_true=y_true)

print(f"BEST VALID SCORE FOR  : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR : {test_score}")

# save tabnet model
saving_path_name = "./tabnet_model_test_1"
saved_filepath = clf.save_model(saving_path_name)

# define new model with basic parameters and load state dict weights
loaded_clf = TabNetRegressor()
loaded_clf.load_model(saved_filepath)

loaded_preds = loaded_clf.predict(X_test)
loaded_test_mse = mean_squared_error(loaded_preds, y_test)

print(f"FINAL TEST SCORE FOR  : {loaded_test_mse}")

assert(test_score == loaded_test_mse)

clf.feature_importances_


explain_matrix, masks = clf.explain(X_test)

from matplotlib import pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(20,20))

for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")
