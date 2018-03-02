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
import lightgbm as lgb
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
from  datetime import datetime as dt

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


MALL_SHOPS = {}


# 返回一个dict,mall:list(shop)
def getMallShopid():
    dfz = pd.read_csv("train_shopinfo.csv")
    for ix, row in dfz.iterrows():
        MALL_SHOPS.setdefault(row['mall_id'], set()).add(row['shop_id'])


getMallShopid()  # MUST RUN ONCE AT XGB BEGIN


# 返回该商场的类别个数
def getNumClass(mall_id):
    return len(MALL_SHOPS[mall_id])


# 0:人流量最多的时间段 1:次多 2:次次多
def change_hour(x):
    if (x == 20) | (x == 18) | (x == 19) | (x == 12) | (x == 13):
        x = 0
    else:
        if (x < 8) | (x == 23):
            x = 1
        else:
            x = 2
    return x


######  经纬度特征处理
MALL_LOC = {}


# 返回一个dict,key是mall_id value是经纬度组成的tuple
def getMallLoc():
    dfff = pd.read_csv("mall_loc.csv")
    for ix, row in dfff.iterrows():
        MALL_LOC[row['mall_id']] = (row['mall_longitude'], row['mall_latitude'])


getMallLoc()  # MUST RUN ONCE AT XGB BEGIN


# 弧度转换
def rad(tude):
    return (math.pi / 180.0) * tude


# (logitude,longitude) 格式
# TODO 需要对异常值进行过滤
def detLongi(mallid, curLongiLatiStr):
    mall_loc = MALL_LOC[mallid]  # 商铺中心点
    longitude1 = mall_loc[0]  # 只取小数部分计算
    latitude1 = mall_loc[1]
    t = curLongiLatiStr.split(',')
    longitude2, latitude2 = float(t[0]), float(t[1])

    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    # latidute相同
    a = 0
    b = rad(longitude1) - rad(longitude2)
    R = 6378137
    detLong = R * 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    return detLong if longitude2 > longitude1 else detLong * (-1)


def detLati(mallid, curLongiLatiStr):
    mall_loc = MALL_LOC[mallid]  # 商铺中心点
    longitude1 = mall_loc[0]  # 只取小数部分计算
    latitude1 = mall_loc[1]
    t = curLongiLatiStr.split(',')
    longitude2, latitude2 = float(t[0]), float(t[1])

    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)

    # longitude相同
    a = radLat1 - radLat2
    b = 0
    R = 6378137
    detLat = R * 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    return detLat if latitude2 > latitude1 else detLat * (-1)


#############################################################
########################  train and predict  ##############################
########################  train and predict  ##############################
#############################################################
def train_predict_write(mallids):
    global TO_FILE
    global DIR_PREFIX
    global DEFAULT_WEIGHT
    # global X
    # global Y
    # global TO_FILE
    for mallid in mallids:
        print("train and predict ", mallid)
        train = pd.read_csv(DIR_PREFIX + "trainDataWithClassIndex/" + mallid + ".csv")

        train_size = len(train)
        maskT = []
        maskV = []
        random.seed(41)
        for i in range(train_size):
            randomNum = random.random()
            if randomNum < 0.81:
                maskT.append(True)
                maskV.append(False)
            else:
                maskT.append(False)
                maskV.append(True)
        # shop_id和类别标号的映射,便于预测结果还原回shop_id
        shop_classIndex_df = pd.concat([train['shop_id'], train['ClassIndex']], axis=1)
        classIndexShopMap = {}
        for ix, row in shop_classIndex_df.iterrows():
            classIndexShopMap[str(row['ClassIndex'])] = row['shop_id']

        ################# 处理Ttrain #################
        print("=" * 50)
        # cur_train_detweight_crosscnt = pd.read_csv('data_fixed/train_detweight_crosscnt/' + mallid + '.csv')
        ## 时间特征
        train['hour'] = train['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        train['hour_split'] = train['hour'].apply(change_hour)  # 给小时分类别0是高峰段...
        train['time_stamp'] = train['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype(
            'int64')  # 日期,多少号
        train['isWeekends'] = train['time_stamp'].apply(
            lambda x: 1 if ((x + 2) % 7 == 0) | ((x + 1) % 7 == 0) else 0)  # 是否周末
        train['num_Week'] = train['time_stamp'].apply(lambda x: (x % 7) + 1)  # 得到星期几
        ## 与商场中wifi的交集个数
        # train['wifi_count'] = train.count(axis=1) - 6 + 1
        ## 重复用户
        train['chongfu_user'] = train.duplicated(['user_id'], keep=False)  # user_id列所有相同的都被标记为重复
        train['chongfu_user'] = train['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        ## 用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        wifi_train = pd.read_csv('wifi_train.csv')
        train = pd.merge(train, wifi_train, how='left', on=['row_id'])
        ## 到每个店铺的强度差以及交集个数
        # train = pd.merge(train, cur_train_detweight_crosscnt, how='left', on=['row_id'])
        # train = train.drop([col for col in train.columns if 'detW_' in col], axis=1)

        ## 经纬度特征
        train['longi_lati'] = train['trade_longitude'].astype("str") + "," + train['trade_latitude'].astype("str")
        train['detLongi'] = train['longi_lati'].apply(lambda x: detLongi(mallid, x))
        train['detLati'] = train['longi_lati'].apply(lambda x: detLati(mallid, x))

        train['distance'] = (train['detLati'] ** 2 + train['detLongi'] ** 2) ** (0.5)
        x = pd.read_csv('data9055/trainData_to_shop_distance/train_' + mallid + '.csv')
        train = pd.merge(train, x, how='left', on=['row_id'])
        # train['distCnt'] = train['distance'] / train['wifi_count']

        ##  画圈加入可能的店铺的类别，平均价格，店铺个数
        # cate_train = pd.read_csv('cate_train_30.csv')
        # train = pd.merge(train, cate_train, how='left', on=['row_id'])
        ## 删除无关属性
        del train['hour'], train['time_stamp'], train['row_id']
        del train['shop_id'], train['user_id']
        del train['longi_lati']

        ################# 处理TEST #################
        test = pd.read_csv(DIR_PREFIX + "testData/" + mallid + ".csv")
        # cate_test = pd.read_csv('cate_test_30.csv')
        # cur_test_detweight_crosscnt = pd.read_csv('data_fixed/test_detweight_crosscnt/' + mallid + '.csv')
        ## 时间特征
        test['hour'] = test['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        test['hour_split'] = test['hour'].apply(change_hour)
        test['time_stamp'] = test['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype('int64')  # 日期
        test['isWeekends'] = test['time_stamp'].apply(lambda x: 1 if ((x % 7 == 2) | (x % 7 == 3)) else 0)  # 是否周末
        test['num_Week'] = test['time_stamp'].apply(lambda x: 7 if ((x + 4) % 7 == 0) else (x + 4) % 7)  # 得到星期几
        ## 与商场中wifi的交集个数
        # test['wifi_count'] = test.count(axis=1) - 6 + 1
        # 重复用户
        test['chongfu_user'] = test.duplicated(['user_id'], keep=False)
        test['chongfu_user'] = test['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        ## 用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        wifi_test = pd.read_csv('wifi_test.csv')
        test = pd.merge(test, wifi_test, how='left', on=['row_id'])
        ## 到每个店铺的强度差以及交集个数
        # test = pd.merge(test, cur_test_detweight_crosscnt, how='left', on=['row_id'])
        # test = test.drop([col for col in test.columns if 'detW_' in col], axis=1)

        ## 经纬度特征
        test['longi_lati'] = test['trade_longitude'].astype("str") + "," + test['trade_latitude'].astype("str")
        test['detLongi'] = test['longi_lati'].apply(lambda x: detLongi(mallid, x))
        test['detLati'] = test['longi_lati'].apply(lambda x: detLati(mallid, x))
        test['distance'] = (test['detLati'] ** 2 + test['detLongi'] ** 2) ** (0.5)
        # test['distCnt'] = test['distance'] / test['wifi_count']
        x = pd.read_csv('data9055/testData_to_shop_distance/test_' + mallid + '.csv')
        test = pd.merge(test, x, how='left', on=['row_id'])
        ## 划圈加入店铺可能的种类,平均价格,可能的店铺总数
        # test = pd.merge(test, cate_test, how='left', on=['row_id'])

        ## 弹出,写入文件用
        test_rowId = test.pop('row_id')
        # 删除无关属性
        del test['hour'], test['time_stamp']
        del test['user_id']  # test没有shopid,rowid前面已经被弹出了
        del test['longi_lati']
        ################# 划分训练集验证集并训练 #################
        print('类别个数', getNumClass(mallid))
        print("rf...")
        Y = train.pop('ClassIndex')
        X = train

        ######################Random Forest####################################
        model = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=None,
                                       min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0.0,
                                       max_features='auto',
                                       max_leaf_nodes=5, bootstrap=True,
                                       oob_score=True, n_jobs=4, random_state=None, verbose=1, warm_start=False,
                                       class_weight=None)
        X = X.fillna(DEFAULT_WEIGHT)
        # Y=Y.fillna(DEFAULT_WEIGHT)
        test = test.fillna(DEFAULT_WEIGHT)
        model.fit(X, Y)
        ################# Predict #################
        # 输出概率
        test_Y = model.predict(test)
        # test_Y = model.predict(test)
        test_Y = pd.concat([test_rowId, pd.Series(test_Y)], axis=1)
        test_Y.columns = ['row_id', 'ClassIndex']
        # print(test_Y)
        # print(type(test_Y.iloc[0,0]))
        # print(type(test_Y.iloc[0,1]))
        shop_id = []
        for index, row in test_Y.iterrows():
            shop_id.append(classIndexShopMap[str(int(row['ClassIndex']))])
        test_Y['shop_id'] = pd.DataFrame(shop_id)
        del test_Y['ClassIndex']

        # 写入结果到文件
        test_Y.to_csv(DIR_PREFIX + "result_rf/" + mallid + "_rf.csv", index=False)
        ###### 只用于 所有mall 预测   #######
    if TO_FILE == 1:
        with open('submit_' + dt.now().strftime("%m_%d_%H_%M") + ".csv", 'w') as f:
            f.write("row_id,shop_id\n")
            for mallid in getMalls():
                df_each = pd.read_csv(DIR_PREFIX + "result_rf/" + mallid + "_rf.csv")
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
# 填充wifi强度的默认值
DEFAULT_WEIGHT = 150
TO_FILE = 1
###################################################################
train_predict_write(getMalls())
# train_predict_write(['m_6803'])
