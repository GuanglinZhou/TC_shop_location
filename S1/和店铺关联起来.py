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
import xgboost as xgb
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
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


getMallShopid()


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


getMallLoc()  ##  !!!! MUST RUN ONCE


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
    detLong = R * 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
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
    detLat = R * 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    return detLat if latitude2 > latitude1 else detLat * (-1)


#############################################################
########################  XGB  ##############################
########################  XGB  ##############################
#############################################################
def train_predict_write(mallids):
    global TO_FILE
    global DIR_PREFIX
    score_file = open("score_" + dt.now().strftime("%m_%d_%H_%M") + ".txt", "w")

    for mallid in mallids:
        print("train and predict ", mallid)
        train = pd.read_csv(DIR_PREFIX + "trainDataWithClassIndex/" + mallid + ".csv")
        train_size = len(train)
        maskT = []
        maskV = []
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
            classIndexShopMap[row['ClassIndex']] = row['shop_id']

        ################# 处理Ttrain #################
        ## 时间特征
        train['hour'] = train['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        train['hour_split'] = train['hour'].apply(change_hour)  # 给小时分类别0是高峰段...
        train['time_stamp'] = train['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype('int64')  # 日期,多少号
        train['isWeekends'] = train['time_stamp'].apply(lambda x: 1 if ((x + 2) % 7 == 0) | ((x + 1) % 7 == 0) else 0)  # 是否周末
        train['num_Week'] = train['time_stamp'].apply(lambda x: (x % 7) + 1)  # 得到星期几
        ## 与商场中wifi的交集个数
        # train['wifi_count'] = train.count(axis=1) - 6
        ## 重复用户
        train['chongfu_user'] = train.duplicated(['user_id'], keep=False)  # user_id列所有相同的都被标记为重复
        train['chongfu_user'] = train['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        ## 用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        # wifi_train = pd.read_csv('wifi_train.csv')
        # train = pd.merge(train, wifi_train, how='left', on=['row_id'])
        ## 经纬度特征
        train['longi_lati'] = train['trade_longitude'].astype("str") + "," + train['trade_latitude'].astype("str")
        train['detLongi'] = train['longi_lati'].apply(lambda x: detLongi(mallid, x))
        train['detLati'] = train['longi_lati'].apply(lambda x: detLati(mallid, x))
        ## 删除无关属性
        del train['hour'], train['time_stamp']
        del train['shop_id'], train['user_id'], train['row_id']
        del train['trade_longitude'], train['trade_latitude'], train['longi_lati']

        ################# 处理TEST #################
        test = pd.read_csv(DIR_PREFIX + "testData/" + mallid + ".csv")
        ## 时间特征
        test['hour'] = test['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        test['hour_split'] = test['hour'].apply(change_hour)
        test['time_stamp'] = test['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype('int64')  # 日期
        test['isWeekends'] = test['time_stamp'].apply(lambda x: 1 if ((x % 7 == 2) | (x % 7 == 3)) else 0)  # 是否周末
        test['num_Week'] = test['time_stamp'].apply(lambda x: 7 if ((x + 4) % 7 == 0) else (x + 4) % 7)  # 得到星期几
        ## 与商场中wifi的交集个数
        # test['wifi_count'] = test.count(axis=1) - 6
        # 重复用户
        test['chongfu_user'] = test.duplicated(['user_id'], keep=False)
        test['chongfu_user'] = test['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        # #用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        # wifi_test = pd.read_csv('wifi_test.csv')
        # test = pd.merge(test, wifi_test, how='left', on=['row_id'])
        ## 经纬度特征
        test['longi_lati'] = test['trade_longitude'].astype("str") + "," + test['trade_latitude'].astype("str")
        test['detLongi'] = test['longi_lati'].apply(lambda x: detLongi(mallid, x))
        test['detLati'] = test['longi_lati'].apply(lambda x: detLati(mallid, x))
        ## 弹出写入文件用
        test_rowId = test.pop('row_id')
        # 删除无关属性
        del test['hour'], test['time_stamp']
        del test['user_id']  # test没有shopid,rowid前面已经被弹出了
        del test['trade_longitude'], test['trade_latitude'], test['longi_lati']

        ################# 划分训练集验证集并训练 #################
        for subSample in ['0.7', '0.8', '0.9', '1']:
            print(subSample)
            params = {
                'booster': 'gbtree',
                'objective': 'multi:softmax',
                'eval_metric': 'merror',
                'num_class': getNumClass(mallid),
                'gamma': 0.1,
                'max_depth': 8,
                'subsample': subSample,
                'colsample_bytree': 0.8,
                'silent': 1,
                'eta': 0.2,
                'seed': 0,
            }
            plst = list(params.items())
            num_rounds = 1500  # 迭代次数
            X_train = train[maskT]
            Y_train = X_train.pop('ClassIndex')
            X_vali = train[maskV]
            Y_vali = X_vali.pop('ClassIndex')
            xgtrain = xgb.DMatrix(X_train, label=Y_train)
            xgval = xgb.DMatrix(X_vali, label=Y_vali)
            watchlist = [(xgtrain, 'train'), (xgval, 'val')]
            model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)

        ## 查看特征重要性
        # feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
        # feat_imp.to_csv('feature_importance_' + dt.now().strftime("%m_%d_%H_%M") + ".txt")
        # score_file.write(mallid + ":" + str(model.best_score) +"\n")

        ################# 预测并构建结果 #################
        test_Y = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        test_Y = pd.concat([test_rowId, pd.Series(test_Y)], axis=1)
        test_Y.columns = ['row_id', 'ClassIndex']
        shop_id = []
        for index, row in test_Y.iterrows():
            shop_id.append(classIndexShopMap[row['ClassIndex']])
        test_Y['shop_id'] = pd.DataFrame(shop_id)
        del test_Y['ClassIndex']

        ####### 只用于 少量mall 测试 #######
        # test_Y.to_csv("tmp/" + mallid + ".csv", index=False)

        # 写入结果到文件
        if TO_FILE == 1:
            test_Y.to_csv(DIR_PREFIX + "result/" + mallid + ".csv", index=False)

    ####### 只用于 所有mall 预测   #######
    score_file.close()
    if TO_FILE == 1:
        with open('submit' + dt.now().strftime("%m_%d_%H_%M") + ".csv", 'w') as f:
            f.write("row_id,shop_id\n")
            for mallid in getMalls():
                df_each = pd.read_csv(DIR_PREFIX + "result/" + mallid + ".csv")
                df_each.to_csv(f, header=False, index=False)
            f.close()


################################################################
SHOP_INFO = {}
SHOP_CATES = list()  # 所有的类别种类


def getShopInfos():
    shopDf = pd.read_csv('train_shopinfo.csv')
    cates = set()
    for ixx, row in shopDf.iterrows():
        SHOP_INFO[row['shop_id']] = (row['longitude'], row['latitude'], row['price'], row['category_id'])
        cates.add(row['category_id'])
    for cate in cates:
        SHOP_CATES.append(cate)


getShopInfos()


# 欧式距离
def getOsDistance(longitude1, latitude1, longitude2, latitude2):
    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    a = radLat1 - radLat2
    b = rad(longitude1) - rad(longitude2)
    R = 6378137
    d = R * 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    return d


###########  生成店铺信息  cate_train.csv  cate_test.csv #################
def genCateTrainAndCateTest():
    # 生成头
    head = "row_id"
    for cate in SHOP_CATES:
        head = head + "," + cate
    head = head + ",avg_price,cates_cnt"

    MALL_SHOPS

    recs = list()
    recs.append(head)

    # mallid = "m_615" # for test
    DIST_1TH = 30
    DIST_2TH = 100
    DIST_3TH = 150

    ##### test集合生成 cate_test.csv ####
    # df = pd.read_csv('data9055/testData/' + mallid + '.csv') # for test
    df = pd.read_csv('test.csv')
    for ix, row in df.iterrows():
        curLongi = row['trade_longitude']
        curLati = row['trade_latitude']
        curRowId = row['row_id']    # test
        mallid = row['mall_id']
        shops = MALL_SHOPS[mallid]

        curRec = str(curRowId)

        # 候选的店铺列表
        dist_1th = list()
        dist_2th = list()
        dist_3th = list()
        dist_all = list()

        for shop in shops:
            shopInfo = SHOP_INFO[shop]
            shopLongi = shopInfo[0]
            shopLati = shopInfo[1]
            dist = getOsDistance(curLongi, curLati, shopLongi, shopLati)
            dist_all.append(dist)
            if dist < DIST_1TH:
                dist_1th.append(shop)
            elif dist < DIST_2TH:
                dist_2th.append(shop)
            elif dist < DIST_3TH:
                dist_3th.append(shop)
        # 暂时只取dist_1th
        # 用wifi交集的个数再过滤掉一些shop

        curCates = set()
        total_price = 0.0
        cates_cnt = 0
        for shop in dist_1th:
            shopInfo = SHOP_INFO[shop]
            shopPrice = shopInfo[2]
            shopCate = shopInfo[3]
            curCates.add(shopCate)
            total_price = total_price + shopPrice
            cates_cnt = cates_cnt + 1
        avg_price = 'NaN'
        if cates_cnt != 0:
            avg_price = total_price / cates_cnt
        for cate in SHOP_CATES:
            if cate in curCates:
                curRec = curRec + "," + "1"
            else:
                curRec = curRec + "," + "0"
        curRec = curRec + "," + str(avg_price) + "," + str(cates_cnt)
        recs.append(curRec)
    out = open("cate_test_" + str(DIST_1TH) + ".csv", 'w')
    for rec in recs:
        out.write(rec + "\n")
    out.close()

    ########## train集合生成 cate_train.csv
    recs = list()
    df = pd.read_csv('train.csv') # for test
    for ix, row in df.iterrows():
        curLongi = row['trade_longitude']
        curLati = row['trade_latitude']
        curRowId = ix  # train
        mallid = row['mall_id']
        shops = MALL_SHOPS[mallid]

        curRec = str(curRowId)

        # 候选的店铺列表
        dist_1th = list()
        dist_2th = list()
        dist_3th = list()
        dist_all = list()

        for shop in shops:
            shopInfo = SHOP_INFO[shop]
            shopLongi = shopInfo[0]
            shopLati = shopInfo[1]
            dist = getOsDistance(curLongi, curLati, shopLongi, shopLati)
            dist_all.append(dist)
            if dist < DIST_1TH:
                dist_1th.append(shop)
            elif dist < DIST_2TH:
                dist_2th.append(shop)
            elif dist < DIST_3TH:
                dist_3th.append(shop)
        # print(len(dist_all), len(dist_1th)) #, len(dist_2th), len(dist_3th))
        # 暂时只取dist_1th

        curCates = set()
        total_price = 0.0
        cates_cnt = 0
        for shop in dist_1th:
            shopInfo = SHOP_INFO[shop]
            shopPrice = shopInfo[2]
            shopCate = shopInfo[3]
            curCates.add(shopCate)
            total_price = total_price + shopPrice
            cates_cnt = cates_cnt + 1
        avg_price = 'NaN'
        if cates_cnt != 0:
            avg_price = total_price / cates_cnt
        for cate in SHOP_CATES:
            if cate in curCates:
                curRec = curRec + "," + "1"
            else:
                curRec = curRec + "," + "0"
        curRec = curRec + "," + str(avg_price) + "," + str(cates_cnt)
        recs.append(curRec)
    out = open("cate_train_" + str(DIST_1TH) + ".csv", 'w')
    for rec in recs:
        out.write(rec + "\n")
    out.close()

# 生成cate_train_xx.csv cate_test_xx.csv
genCateTrainAndCateTest()