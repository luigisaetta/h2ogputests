import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import time

%%time 
# Fetch dataset using sklearn cov = fetch_covtype() 
X = cov.data 
y = cov.target

%%time
# Create 0.75/0.25 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=42)

%%time
# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

num_round = 10
maxdepth = 6
# base parameters
param = {'tree_method': 'gpu_hist',
         'grow_policy': 'depthwise',
         'max_depth': maxdepth,
         'random_state': 1234,
         'objective': 'multi:softmax', # Specify multiclass classification
         'num_class': 8, # Number of possible output classes
         'base_score': 0.5,
         'booster': 'gbtree',
         'colsample_bylevel': 1,
         'colsample_bytree': 1,
         'gamma': 0,
         'learning_rate': 0.1, 
         'max_delta_step': 0,
         'min_child_weight': 1,
         'missing': None,
         'n_estimators': 3,
         'scale_pos_weight': 1,
         'silent': True,
         'subsample': 1,
         'verbose': True,
         'n_jobs': -1
         }

%%time
# First setup: GPU HIST DEPTHWISE
param['tree_method'] = 'gpu_hist'
param['grow_policy'] = 'depthwise'
param['max_depth'] = maxdepth
param['max_leaves'] = 0
gpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

%%time
# Second setup: GPU HIST LOSSGUIDE
param['tree_method'] = 'gpu_hist'
param['grow_policy'] = 'lossguide'
param['max_depth'] = 0
param['max_leaves'] = np.power(2,maxdepth)
gpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)
print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

%%time
# Third setup: CPU HIST DEPTHWISE
param['tree_method'] = 'hist'
param['grow_policy'] = 'depthwise'
param['max_depth'] = maxdepth
param['max_leaves'] = 0
cpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))

%%time
# Fourth setup: CPU HIST LOSSGUIDE
param['tree_method'] = 'hist'
param['grow_policy'] = 'lossguide'
param['max_depth'] = 0
param['max_leaves'] = np.power(2,maxdepth)
cpu_res = {} # Store accuracy result
tmp = time.time()
# Train model
xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=cpu_res)
print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))
