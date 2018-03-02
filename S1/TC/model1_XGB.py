# -*-coding:utf-8 -*-
import os
import math
import re
import uuid
import operator
import random
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from  datetime import datetime as dt


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


def init_static():
    getMallShopid()  # MUST RUN ONCE AT XGB BEGIN
    getMallLoc()  # MUST RUN ONCE AT XGB BEGIN


#############################################################
########################  XGB  ##############################
########################  XGB  ##############################
#############################################################
def train_predict_write(mallids):
    global DIR_PREFIX

    for mallid in mallids:
        print("train and predict ", mallid)
        train = pd.read_csv(DIR_PREFIX + "trainDataWithClassIndex/" + mallid + ".csv")

        ################# 处理Ttrain #################
        print("=" * 100)
        ## 时间特征
        train['hour'] = train['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        # train['hour_split'] = train['hour'].apply(change_hour)  # 给小时分类别0是高峰段...
        # train['time_stamp'] = train['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype('int64')  # 日期,多少号
        # train['isWeekends'] = train['time_stamp'].apply(lambda x: 1 if ((x + 2) % 7 == 0) | ((x + 1) % 7 == 0) else 0)  # 是否周末
        # train['num_Week'] = train['time_stamp'].apply(lambda x: (x % 7) + 1)  # 得到星期几
        ## 与商场中wifi的交集个数
        # train['wifi_count'] = train.count(axis=1) - 6 + 1
        ## 重复用户
        # train['chongfu_user'] = train.duplicated(['user_id'], keep=False)  # user_id列所有相同的都被标记为重复
        # train['chongfu_user'] = train['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        ## 用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        wifi_train = pd.read_csv('wifi_train.csv')
        train = pd.merge(train, wifi_train, how='left', on=['row_id'])
        del train['max_strength'], train['connect_strength']
        ## 到每个店铺的强度差以及交集个数
        cur_train_detweight_crosscnt = pd.read_csv('data_fixed/train_detweight_crosscnt/' + mallid + '.csv')
        train = pd.merge(train, cur_train_detweight_crosscnt, how='left', on=['row_id'])
        train = train.drop([col for col in train.columns if col.startswith('detW_')], axis=1)
        train = train.drop([col for col in train.columns if col.startswith('cnt_')], axis=1)
        print(train.columns)
        ## 经纬度特征
        train['longi_lati'] = train['trade_longitude'].astype("str") + "," + train['trade_latitude'].astype("str")
        train['detLongi'] = train['longi_lati'].apply(lambda x: detLongi(mallid, x))
        train['detLati'] = train['longi_lati'].apply(lambda x: detLati(mallid, x))
        # train['distance'] = (train['detLati'] ** 2 + train['detLongi'] ** 2) ** (0.5)
        # train['distCnt'] = train['distance'] / train['wifi_count']
        ## 删除无关属性
        del train['hour'], train['time_stamp'], train['row_id']
        del train['shop_id'], train['user_id']
        del train['longi_lati']

        ################# 处理TEST #################
        test = pd.read_csv(DIR_PREFIX + "testData/" + mallid + ".csv")
        ## 时间特征
        test['hour'] = test['time_stamp'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype('int64')  # 小时
        # test['hour_split'] = test['hour'].apply(change_hour)
        # test['time_stamp'] = test['time_stamp'].apply(lambda x: x.split(' ')[0].split('-')[2]).astype('int64')  # 日期
        # test['isWeekends'] = test['time_stamp'].apply(lambda x: 1 if ((x % 7 == 2) | (x % 7 == 3)) else 0)  # 是否周末
        # test['num_Week'] = test['time_stamp'].apply(lambda x: 7 if ((x + 4) % 7 == 0) else (x + 4) % 7)  # 得到星期几
        ## 与商场中wifi的交集个数
        # test['wifi_count'] = test.count(axis=1) - 6 + 1
        # 重复用户
        # test['chongfu_user'] = test.duplicated(['user_id'], keep=False)
        # test['chongfu_user'] = test['chongfu_user'].apply(lambda x: 1 if (x == True) else 0)
        ## 用户的最强wifi强度、用户有无连接以及连接时候的wifi强度
        wifi_test = pd.read_csv('wifi_test.csv')
        test = pd.merge(test, wifi_test, how='left', on=['row_id'])
        del test['max_strength'], test['connect_strength']
        ## 到每个店铺的强度差以及交集个数
        cur_test_detweight_crosscnt = pd.read_csv('data_fixed/test_detweight_crosscnt/' + mallid + '.csv')
        print(test.shape)
        test = pd.merge(test, cur_test_detweight_crosscnt, how='left', on=['row_id'])
        test = test.drop([col for col in test.columns if col.startswith('detW_')], axis=1)
        print(test.shape)
        test = test.drop([col for col in test.columns if col.startswith('cnt_')], axis=1)
        print(test.shape)
        ## 经纬度特征
        test['longi_lati'] = test['trade_longitude'].astype("str") + "," + test['trade_latitude'].astype("str")
        test['detLongi'] = test['longi_lati'].apply(lambda x: detLongi(mallid, x))
        test['detLati'] = test['longi_lati'].apply(lambda x: detLati(mallid, x))
        # test['distance'] = (test['detLati'] ** 2 + test['detLongi'] ** 2) ** (0.5)
        # test['distCnt'] = test['distance'] / test['wifi_count']
        ## 弹出,写入文件用
        test_rowId = test.pop('row_id')
        # 删除无关属性
        del test['hour'], test['time_stamp']
        del test['user_id']  # test没有shopid,rowid前面已经被弹出了
        del test['longi_lati']

        ################# 划分训练集验证集并训练 #################
        print('类别个数', getNumClass(mallid))
        params = {
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'eval_metric': 'merror',
            'num_class': getNumClass(mallid),
            'max_depth': 8,
            'lambda': 1,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1,
            'eta': 0.05,
        }
        plst = list(params.items())

        Y = train.pop('ClassIndex')
        X = train
        xgtrain = xgb.DMatrix(X, label=Y)
        watchlist = [(xgtrain, 'train')]
        # cv_results = xgb.cv(plst, xgtrain, num_boost_round=300, metrics='merror', nfold=5, early_stopping_rounds=30)
        # print(cv_results)

        model = xgb.train(plst, xgtrain, 300, watchlist, early_stopping_rounds=30)

        # Valid用于找参数
        # X_train = train[maskT]
        # Y_train = X_train.pop('ClassIndex')
        # X_vali = train[maskV]
        # Y_vali = X_vali.pop('ClassIndex')
        # xgtrain = xgb.DMatrix(X_train, label=Y_train)
        # xgval = xgb.DMatrix(X_vali, label=Y_vali)
        # watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        # model = xgb.train(plst, xgtrain, 300,  watchlist, early_stopping_rounds=30)

        ## XGB查看特征重要性
        # feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
        # feat_imp.to_csv('feature_importance_' + dt.now().strftime("%m_%d_%H_%M") + ".txt")
        # score_file.write(mallid + ":" + str(model.best_score) +"\n")

        ################# 预测并构建结果 #################
        # 输出概率结果
        print('model.best_ntree_limit:', model.best_ntree_limit)
        test_Y = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        probStrs = list()
        for pyOne in test_Y:
            curStr = ','.join(map(str, pyOne))
            probStrs.append(curStr)
        test_Y = pd.concat([test_rowId, pd.Series(probStrs)], axis=1)
        test_Y.columns = ['row_id', 'class_prob']
        # 写入概率结果到文件
        test_Y.to_csv('model_output_prob/' + mallid + "_m1.csv", index=False)

        # 输出类别结果
        test_Y = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        test_Y_classIndex = list()
        for each_y in test_Y:
            max_index, max_value = max(enumerate(each_y), key=operator.itemgetter(1))
            test_Y_classIndex.append(max_index)
        test_Y = pd.concat([test_rowId, pd.Series(test_Y_classIndex)], axis=1)
        test_Y.columns = ['row_id', 'ClassIndex']
        classIndexShopMap = json.load(open('data_fixed/classindex_shop/' + mallid + "_ClassIndexMap.txt", 'r'))
        shop_id = []
        for index, row in test_Y.iterrows():
            shop_id.append(classIndexShopMap[str(int(row['ClassIndex']))])
        test_Y['shop_id'] = pd.DataFrame(shop_id)
        del test_Y['ClassIndex']
        # 写入类别结果到文件
        test_Y.to_csv("model_output_class/" + mallid + "_m1.csv", index=False)


################################################
################################################
# 必须 / 结尾
DIR_PREFIX = "./data9055/"
# 初始化全局的资源(比如MALL_SHOP表)
init_static()
# 分多个电脑跑
mallsum = pd.read_csv('train_mall.csv')
mall1 = np.array(mallsum['mall_id'].iloc[:33])
mall2 = np.array(mallsum['mall_id'].iloc[33:])
###################################################################
train_predict_write(getMalls() - {'m_979', 'm_1621', 'm_9068', 'm_5076', 'm_2907', 'm_909', 'm_5154', 'm_690', 'm_1175', 'm_6587', 'm_5352', 'm_7994', 'm_3005', 'm_2878', 'm_7523', 'm_5529', 'm_4572', 'm_2123', 'm_4187', 'm_3871', 'm_3019', 'm_2009', 'm_822', 'm_4121', 'm_5810', 'm_4094', 'm_4543', 'm_1263', 'm_4759', 'm_4079', 'm_2415', 'm_4923', 'm_4828', 'm_3916', 'm_1831', 'm_1021', 'm_3739', 'm_4459', 'm_7800', 'm_1375', 'm_4011', 'm_3517'}
)  # run on ALL data
# train_predict_write(['m_6803'])  # m_6803 for quick test


############  用于合并分类结果文件到一个文件然后提交 #######
# dir是各个文件所在的目录
def merge_files():
    with open('submit记录/xgb_' + dt.now().strftime("%m_%d_%H_%M") + ".csv", 'w') as f:
        f.write("row_id,shop_id\n")
        for mallid in getMalls():
            df_each = pd.read_csv('model_output_class/' + mallid + '_m1.csv')
            df_each.to_csv(f, header=False, index=False)
        f.close()
merge_files()
