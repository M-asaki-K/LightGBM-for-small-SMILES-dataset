# データ処理ライブラリ
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import math
# データ可視化ライブラリ
import matplotlib.pyplot as plt

# LightGBM
import xgboost as xgb
import lightgbm as lgb
# Scikit-learn（評価算出）
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold,StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import seaborn as sns

#学習データとテストデータのインポート（https://github.com/Saket-Uoft/Artificial-Intelligence-for-Predicting-Materials-Properties）
#mordred記述子を使用、3Dは100回conformerを生成しaugment
df = pd.read_csv(r'D:\OneDrive\デスクトップ\Data science\JACI\JACI202012\dataTg_new_train.csv')
dft = pd.read_csv(r'D:\OneDrive\デスクトップ\Data science\JACI\JACI202012\dataTg_new_test.csv')

#説明変数と目的変数の抽出
x_train = df[df.columns[2:]]
x_train['type'] = 1
y_train = df[df.columns[1]]
y_train
x_test = dft[dft.columns[2:]]
x_test['type'] = 1
y_test =dft[dft.columns[1]]

#GroupKfold（trainデータの一部を検証用に設定しパラメータチューニング）用に、
#化合物ごとにデータをグルーピング（Augmentした同じ化合物は目的変数も同一なので、それをグループのタグに利用）
groups = np.array(y_train) 
groups

row, col = x_train.shape

#lightgbmモデルの定義
def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data（検証用データのこと）
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds（ここではGroupkfold）
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, groups)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

#GroupKfoldのfold数設定
n_fold = 5
folds = GroupKFold(n_splits=n_fold)

#ベイズ最適化の目的関数（lightGBMにて得られる、交差検証の予測精度）
def lgb_cv(num_le, max_dep,lr,bag_seed):
    #Set hyper parameters（ここは調整代がまだあります。今回は4変数に設定）
    params = {'num_leaves': int(round(num_le)),
              #'eta' : 0.001,
              #'min_child_weigh': 
              #"lambda" : ai,
              #'min_child_samples': 32,
              'objective': 'regression',  #I will check it.
              'max_depth': int(round(max_dep)),
              'learning_rate': lr,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.8,
              "bagging_seed": int(round(bag_seed)),
              "metric": 'mae',
              "verbosity": -1,
              #'reg_alpha': 0.8,
              #'reg_lambda': 0.2,
              #'colsample_bytree': 0.6,
              'lambda_l1': 0.8,
              'lambda_l2': 0.2,
              'feature_fraction': 0.6,
              }

    result_dict_lgb_oof = train_model_regression(X=x_train, 
                                             X_test=x_test, 
                                             y=y_train, 
                                             params=params, 
                                             folds=folds,
                                             model_type='lgb',
                                             eval_metric='group_mae',
                                             plot_feature_importance=False,
                                             verbose=30000, 
                                             early_stopping_rounds=200, 
                                             n_estimators=30000)
                                             
    sc = ((result_dict_lgb_oof['scores'])[0] + (result_dict_lgb_oof['scores'])[1]) / 2*(-1)
    return sc

#ベイズ最適化の設定
rf_cv_bo = BayesianOptimization(
    lgb_cv, # 上の関数を与える

    {'num_le': (5, 100), # (min,max)でパラメータの探索範囲を指定
     'max_dep': (3, 100),
     'lr': (0.00001, 0.2),
     'bag_seed' : (3, 100),
    },
    verbose=2, # 0だと学習過程を表示をしない、デフォルトの2なら全て表示する、1は最高値が更新されたら表示
    random_state=0
)
#ベイズ最適化を実行（初期サンプル、ベイズ最適化の試行回数、試行にあたっての獲得関数）
rf_cv_bo.maximize(init_points=20, n_iter=30, acq='ei')
rf_cv_bo.res

#最適化されたパラメータの読込
num_le = rf_cv_bo.max['params']['num_le']
max_dep = rf_cv_bo.max['params']['max_dep']
lr = rf_cv_bo.max['params']['lr']
bag_seed = rf_cv_bo.max['params']['bag_seed']

#最高成績の確認
rf_cv_bo.max

#最適化した条件にて再学習：Set hyper parameters
params = {'num_leaves': int(round(num_le)),
          #'eta' : 0.001,
          #'min_child_weigh': 
          #"lambda" : ai,
          #'min_child_samples': 32,
          'objective': 'regression',  #I will check it.
          'max_depth': int(round(max_dep)),
          'learning_rate': lr,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "bagging_seed": int(round(bag_seed)),
          "metric": 'mae',
          "verbosity": -1,
          #'reg_alpha': 0.8,
          #'reg_lambda': 0.2,
          #'colsample_bytree': 0.6,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.6,
          }

result_dict_lgb_oof = train_model_regression(X=x_train, 
                                         X_test=x_test, 
                                         y=y_train, 
                                         params=params, 
                                         folds=folds,
                                         model_type='lgb',
                                         eval_metric='group_mae',
                                         plot_feature_importance=False,
                                         verbose=60000, 
                                         early_stopping_rounds=200, 
                                         n_estimators=60000)
                                         
sc = ((result_dict_lgb_oof['scores'])[0] + (result_dict_lgb_oof['scores'])[1]) / 2*(-1)

#最適化したモデルにおける交差検証結果の出力
predd = result_dict_lgb_oof['prediction']

#交差検証結果と、テストデータに対する予測―実測結果の比較
#（分布が一致している＝過学習しておらず、テストデータはモデルの適用範囲内である）
plt.plot(y_test, predd, marker = '+', ls = 'None', label = "test")
plt.plot(y_train, result_dict_lgb_oof['oof'], marker = '.', ls = 'None', label = "train")
plt.xlabel("Observed value")
plt.ylabel("Predicted Value")
plt.axis([-3,3,-3,3])
plt.axes().set_aspect('equal')
