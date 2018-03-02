import os
import re
import uuid
import operator
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import matplotlib.pyplot as plt
# 要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns
import xgboost as xgb
import sklearn.preprocessing as preprocessing

from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
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

# 控制matplot的中文显示,使用时用u"中文字符"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 控制pandas在终端的显示
pd.set_option('display.height', 10000)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 20000)
import datetime
import time


##################################################################
# 返回一个set 里面包含所有的mall_id值
def getMalls():
    r = set()
    df = pd.read_csv("train_shopinfo.csv")
    malls = df['mall_id'].values
    for mall in malls:
        r.add(mall)
    return r


# 得到mall:list(shop)集合
def getMallShopid():
    mall_shop = {}
    df = pd.read_csv("train_shopinfo.csv")
    for ix, row in df.iterrows():
        mall_shop.setdefault(row['mall_id'], set()).add(row['shop_id'])
    # for (k,v) in mall_shop.items():
    #     print(k, v)
    return mall_shop


def getShopLoc():
    shop_loc = {}
    df = pd.read_csv("shop_loc.csv")
    for ix, row in df.iterrows():
        shop_loc[row['shop_id']] = (row['shop_longitude'], row['shop_latitude'])
    return shop_loc


def getShopWifiWeight():
    shop_wifi_weight = {}
    df1 = pd.read_csv('H:/TianChi/data/shop_wifi_weight.xk', sep='=', header=None)
    df1.columns = ["shop_id", 'wifi_inthis']
    for index, row in df1.iterrows():
        # print len(eval(row.wifi_inthis))
        shop_wifi_weight[row.shop_id] = eval(row.wifi_inthis)
    return shop_wifi_weight


def getWifiWeight(wifi_infos):
    wifi_weight = {}
    infos = wifi_infos.split(";")
    for info in infos:
        parts = info.split("|")
        wifi_weight[parts[0]] = abs(int(parts[1]))
    return wifi_weight



####################################################################################################
###########################构造wifi特征,然后写入csv文件,便于后续多分类##################################
###########################构造wifi特征,然后写入csv文件,便于后续多分类##################################
###########################构造wifi特征,然后写入csv文件,便于后续多分类##################################
####################################################################################################



# filteredSorted_wifi_cnt:经过筛选的有序的[wifi:cnt],filteredSorted_wifi_cnt[0]是wifi名字,filteredSorted_wifi_cnt[1]是该wifi出现的次数
# wifi_infos:一个string,格式是:bsssid|强度值|true;bsssid|强度值|false;.....
# 返回一个list,wifi强度列表,和特征顺序的wifi对应
def getWeightList(filteredSorted_wifi_cnt, wifi_infos):
    weight = list()
    wifi_weight = getWifiWeight(wifi_infos)
    for wifi_cnt in filteredSorted_wifi_cnt:
        wifi = wifi_cnt[0]  # 按照特征顺序的wifi名字
        weight.append(wifi_weight.get(wifi, DEFAULT_WEIGHT))
    return weight


# mallid:根据mallid定位到该mallid对应的csv文件
# 返回一个tuple,tuple[0]是按照wifi出现次数降序排列的一个list,tuple[0][0]是wifi名字,tuple[0][1]是该wifi出现的次数
#    tuple[1]是一个dict,{wifiname:cnt}
def getWifiCnt(mallid):
    wifi_cnt = {}
    data = pd.read_csv("./data_fixed/trainsplits/" + mallid + ".csv")
    for ix, row in data.iterrows():
        wifi_infos = row['wifi_infos'].split(";")
        for info in wifi_infos:
            parts = info.split("|")
            wifi_cnt[parts[0]] = wifi_cnt.get(parts[0], 0) + 1
    sorted_wifi_cnt = sorted(wifi_cnt.items(), key=lambda x: x[1], reverse=True)
    # for wifiCnt in sorted_wifi_cnt:
    #     print(wifiCnt[0], wifiCnt[1]) #wifi, wifi对应的次数
    return sorted_wifi_cnt, wifi_cnt  # 第一个值有序,第二个值是dict形式


# sorted_wifi_cnt: 一个list,每个元素是一个tuple,sorted_wifi_cnt[0][0]是第一个元素的wifi名字,sorted_wifi_cnt[0][1]是第一个元素wifi对应的出现次数
# 返回一个list,每个元素是一个tuple,结构类似sorted_wifi_cnt,不过去掉了出现次数太少的元素
def getFilterdWifiCnt(sorted_wifi_cnt):
    global THRES_HOLD_CNT
    r = list()
    for wifi_cnt in sorted_wifi_cnt:
        if wifi_cnt[1] > THRES_HOLD_CNT:
            r.append(wifi_cnt)
    return r


# filteredSorted_wifi_cnt: 经过筛选的有序的[wifi:cnt],filteredSorted_wifi_cnt[0]是wifi名字,filteredSorted_wifi_cnt[1]是该wifi出现的次数
# 返回一个string,特征名字,要写入csv文件的表头
def constructTrainHead(filteredSorted_wifi_cnt):
    head = "row_id,user_id,shop_id," + "time_stamp," + "isWeek," + "trade_longitude," + "trade_latitude"
    for t in filteredSorted_wifi_cnt:
        wifiName = t[0]
        head = head + "," + wifiName
    # print("new csv head is:", head)
    return head


# 用于训练集构造
# mallid: 根据mallid定位到该mallid对应的csv文件
# filteredSorted_wifi_cnt: 经过筛选的有序的[wifi:cnt],filteredSorted_wifi_cnt[0]是wifi名字,filteredSorted_wifi_cnt[1]是该wifi出现的次数
# 返回一个List(包含表头),每个元素都是csv文件的一行
def constructTrainNewRecord(mallid, filteredSorted_wifi_cnt):
    newRecs = list()

    head = constructTrainHead(filteredSorted_wifi_cnt)
    newRecs.append(head)

    data = pd.read_csv("./data_fixed/trainsplits/" + mallid + ".csv")
    for ix, row in data.iterrows():
        curShopId = row['shop_id']
        userId = row['user_id']
        rowId = row['row_id']
        curTimeStamp = row['time_stamp']
        isWeek = 1 if datetime.datetime.strptime(curTimeStamp, '%Y-%m-%d %H:%M').weekday() >= 5 else 0
        curTradeLogi = row['trade_longitude']
        curTradeLati = row['trade_latitude']
        curWifiInfos = row['wifi_infos']
        # 得到与特征顺序一致的wifi强度
        weightList = getWeightList(filteredSorted_wifi_cnt, curWifiInfos)
        rec = str(rowId) + "," + userId + "," +curShopId + "," + curTimeStamp + "," + str(isWeek) + "," + str(curTradeLogi) + "," + str(curTradeLati)
        for w in weightList:  # 默认值暂时空着,先设置为DEFAULT_WEIGHT作为一个标记
            if w == DEFAULT_WEIGHT:
                rec = rec + ","
            else:
                rec = rec + "," + str(w)
        newRecs.append(rec)
    return newRecs


def writeNewTrainToFile(mallids):
    global DIR_PREFIX
    for mallid in mallids:
        out = open(DIR_PREFIX + "trainData/" + mallid + ".csv", "w")
        lines = constructTrainNewRecord(mallid, getFilterdWifiCnt(getWifiCnt(mallid)[0]))
        for line in lines:
            out.write(line + "\n")
        out.close()


# 测试集的头
def constructTestHead(filteredSorted_wifi_cnt):
    head = 'user_id,row_id,' + "time_stamp,isWeek," + "trade_longitude," + "trade_latitude"
    for t in filteredSorted_wifi_cnt:
        wifiName = t[0]
        head = head + "," + wifiName
    # print("new csv head is:", head)
    return head


def constructTestNewRecord(mallid):
    global DEFAULT_WEIGHT
    newRecs = list()
    filteredSorted_wifi_cnt = getFilterdWifiCnt(getWifiCnt(mallid)[0])
    head = constructTestHead(filteredSorted_wifi_cnt)
    newRecs.append(head)  # 测试集多一个row_id列名

    data = pd.read_csv("./data_fixed/testsplits/" + mallid + ".csv")
    for ix, row in data.iterrows():
        curRowId = row['row_id']
        userId = row['user_id']
        curTimeStamp = row['time_stamp']
        isWeek = 1 if datetime.datetime.strptime(curTimeStamp, '%Y-%m-%d %H:%M').weekday() >= 5 else 0
        curTradeLogi = row['trade_longitude']
        curTradeLati = row['trade_latitude']
        curWifiInfos = row['wifi_infos']
        # 得到与特征顺序一致的wifi强度
        weightList = getWeightList(filteredSorted_wifi_cnt, curWifiInfos)
        rec = userId + "," +str(curRowId) + "," + curTimeStamp + "," + str(isWeek) + "," + str(curTradeLogi) + "," + str(curTradeLati)
        for w in weightList:  # 默认值暂时空着
            if w == DEFAULT_WEIGHT:
                rec = rec + ","
            else:
                rec = rec + "," + str(w)
        newRecs.append(rec)
    return newRecs


# mallid:对该商铺进行特征构造后,写入文件
# 把经过特征构造后的新数据写入到新的文件
def writeNewTestToFile(mallids):
    global DIR_PREFIX
    for mallid in mallids:
        out = open(DIR_PREFIX + "testData/" + mallid + ".csv", "w")
        lines = constructTestNewRecord(mallid)

        for line in lines:
            out.write(line + "\n")
        out.close()


# shops,一个set,shop集合
# 返回一个dict,key是shopid,value是其对应的类别标号,从0开始
def getShopClassDict(shops):
    classIndex = 0
    shop_classIndex = {}
    for shop in shops:
        shop_classIndex[shop] = classIndex
        classIndex = classIndex + 1
    return shop_classIndex


# 构造完成后打上类别索引
def trans2ClassIndex(mallids):
    global DIR_PREFIX
    for mallid in mallids:
        shops = getMallShopid()[mallid]
        shop_classIndex = getShopClassDict(shops)

        dfTrain = pd.read_csv(DIR_PREFIX + "trainData/" + mallid + ".csv")
        print('read complete')
        dfTrain['ClassIndex'] = -1

        for ix, row in dfTrain.iterrows():
            classIndex = shop_classIndex[row['shop_id']]
            dfTrain.set_value(ix, 'ClassIndex', classIndex)
        dfTrain.to_csv(DIR_PREFIX + "trainDataWithClassIndex/" + mallid + ".csv", index=False)


###########################################################################################
########################只在一台电脑上构造###################################################
########################只在一台电脑上构造###################################################
########################只在一台电脑上构造###################################################
###########################################################################################

# 需要调节的参数
# wifi强度默认值,用于标识是否存在该wifi,便于写入csv判断
DEFAULT_WEIGHT = 200    # 不需要修改了(200并没有出现过)
# 取大于x个出现次数的wifi名字做特征
THRES_HOLD_CNT = 5

######  目录结构必须如下 ######
# DIR_PREFIX
#    result/
#    trainData/
#    testData/
#    trainDataWithClassIndex/
DIR_PREFIX = "./data9055_5th/"  # 必须 / 结尾


##### 训练集构造数据   #####
mallids = getMalls()  # mallids可以任何一个可以迭代的东西
writeNewTrainToFile(mallids)
# 测试集构造数据
writeNewTestToFile(mallids)
# 打上类别索引
trans2ClassIndex(mallids)





