# -*-coding:utf-8 -*-
import os
import math
import re
import uuid
import operator
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns
# import xgboost as xgb
# import lightgbm as lgb
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from datetime import datetime as dt

# 控制matplot的中文显示,使用时用u"中文字符"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 控制pandas在终端的显示
pd.set_option('display.height', 10000)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 20000)


# 返回一个set 里面包含所有的mall_id值
def getMalls():
    r = set()
    df = pd.read_csv("train_shopinfo.csv")
    malls = df['mall_id'].values
    for mall in malls:
        r.add(mall)
    return r


#############################################################
def model_merge_write(mallids,  xgb_coe, rf_coe):
    global TO_FILE
    global DIR_PREFIX
    global DEFAULT_WEIGHT
    # global X
    # global Y
    # global TO_FILE
    for mallid in mallids:
        print("Merging ", mallid, "...")
        train = pd.read_csv(
            DIR_PREFIX + "trainDataWithClassIndex/" + mallid + ".csv")
        # MERGE_MALL = 'data9055/model_file/model_merge/' + mallid + '.csv'
        # merge_mall = open(MERGE_MALL, 'w')
        # merge_mall.write('row_id,shop_id\n')
        # shop_id和类别标号的映射,便于预测结果还原回shop_id
        shop_classIndex_df = pd.concat(
            [train['shop_id'], train['ClassIndex']], axis=1)
        classIndexShopMap = {}
        for ix, row in shop_classIndex_df.iterrows():
            classIndexShopMap[str(row['ClassIndex'])] = row['shop_id']
        #######################################################################
        test = pd.read_csv(DIR_PREFIX + "testData/" + mallid + ".csv")
        # 弹出,写入文件用
        test_rowId = test.pop('row_id')

        ############################# Merge ###################################
        # xgb_data = pd.read_csv('data9055/model_file/model1/' + mallid + '.csv') * xgb_coe
        # rf_data = pd.read_csv('data9055/model_file/model2/' + mallid + '_model2.csv') * rf_coe
        # lgb_data = pd.read_csv('data9055/model_file/model3/' + mallid + '_model3.csv') * lgb_coe
        xgb_data = pd.read_csv('data9055/model_file/model1/' + mallid + '.csv')
        rf_data = pd.read_csv(
            'data9055/model_file/model2/' + mallid + '_model2.csv')
        # lgb_data = pd.read_csv('data9055/model_file/model3/' + mallid + '_model3.csv')
        max_index_list = []
        for i in range(xgb_data.shape[0]):
            xgb_value_list = np.asarray(
                [float(i) for i in xgb_data.iloc[i, 1].split(",")]) * xgb_coe
            rf_value_list = np.asarray(
                [float(i) for i in rf_data.iloc[i, 1].split(",")]) * rf_coe
            # lgb_value_list = np.asarray([float(i) for i in lgb_data.iloc[i, 1].split(",")]) * lgb_coe
            xgb_max_value = np.max(xgb_value_list)
            xgb_max_index = np.argmax(xgb_value_list)
            rf_max_value = np.max(rf_value_list)
            rf_max_index = np.argmax(rf_value_list)
            # lgb_max_value = np.max(lgb_value_list)
            # lgb_max_index = np.argmax(lgb_value_list)
            max_value = max(xgb_max_value, rf_max_value)
            if (xgb_max_value == max_value):
                max_index = xgb_max_index
            elif (rf_max_value == max_value):
                max_index = rf_max_index
            max_index_list.append(max_index)

        result_y = pd.concat([test_rowId, pd.Series(max_index_list)], axis=1)
        result_y.columns = ['row_id', 'ClassIndex']
        shop_id = []
        # print(result_y)
        for index, row in result_y.iterrows():
            shop_id.append(classIndexShopMap[str(row['ClassIndex'])])
        result_y['shop_id'] = pd.DataFrame(shop_id)
        del result_y['ClassIndex']

        # 写入结果到文件
        result_y.to_csv(DIR_PREFIX + "model_file/model_merge/" +
                        mallid + "_xgb_rf.csv", index=False)
        ####### 只用于 所有mall 预测   #######
    if TO_FILE == 1:
        with open('submit_' + dt.now().strftime("%m_%d_%H_%M") + "_merge.csv", 'w') as f:
            f.write("row_id,shop_id\n")
            for mallid in getMalls():
                df_each = pd.read_csv(
                    DIR_PREFIX + "model_file/model_merge/" + mallid + "_xgb_rf.csv")
                df_each.to_csv(f, header=False, index=False)
            f.close()


######  目录结构必须如下 ######
######  目录结构必须如下 ######
######  目录结构必须如下 ######
# DIR_PREFIX
#    result/
#    trainData/
#    trainData/
#    trainData/
#    testData/
#    trainDataWithClassIndex/
DIR_PREFIX = "./data9055/"  # 必须 / 结尾
mallsum = pd.read_csv('train_mall.csv')
mall1 = np.array(mallsum['mall_id'].iloc[:33])
mall2 = np.array(mallsum['mall_id'].iloc[33:])
TO_FILE = 1
###################################################################
# model_merge_write(getMalls(), 0.5, 0.3, 0.2)  # tengyuan
# model_merge_write(getMalls(), 0.4, 0.3, 0.3)  # memeda
# model_merge_write(getMalls(), 0.6, 0.2, 0.2)  # fangshu
# model_merge_write(getMalls(), 0.34, 0.33, 0.33)
# model_merge_write(['m_6803'], 0.5, 0.3, 0.2)
model_merge_write(getMalls(), 0.6, 0.4)
