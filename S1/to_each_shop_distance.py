# -*-coding:utf-8 -*-
##增加每条记录到每个shop的距离
import numpy as np
import pandas as pd
import math


# 返回一个set 里面包含所有的mall_id值
def getMalls():
    r = set()
    df = pd.read_csv("train_shopinfo.csv")
    malls = df['mall_id'].values
    for mall in malls:
        r.add(mall)
    return r


shopid_long_lat = {}


# 返回一个dict,里面包含shopid和其对应的经纬度字典，shopid作为key,经纬度作为值
def getShopid_longlat_dic():
    df = pd.read_csv("train_shopinfo.csv")
    for index, row in df.iterrows():
        shopid_long_lat[row.shop_id] = []
        shopid_long_lat[row.shop_id].append(row.longitude)
        shopid_long_lat[row.shop_id].append(row.latitude)


MALL_SHOPS = {}


# 返回一个dict,mall:list(shop)
def getMallShopid():
    dfz = pd.read_csv("train_shopinfo.csv")
    for ix, row in dfz.iterrows():
        MALL_SHOPS.setdefault(row['mall_id'], set()).add(row['shop_id'])


# 弧度转换
def rad(tude):
    return (math.pi / 180.0) * tude


# 欧式距离
def produceLocationInfo(latitude1, longitude1, latitude2, longitude2):
    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    a = radLat1 - radLat2
    b = rad(longitude1) - rad(longitude2)
    R = 6378137
    d = R * 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    detallat = abs(a) * R
    return round(d)


#################################
getShopid_longlat_dic()
getMallShopid()
for mall_id in getMalls():
# for mall_id in ["m_7800"]:
    print "正在写入"+mall_id+"的训练集"
    traindata = pd.read_csv("data9055/trainData/" + mall_id + ".csv")
    file = open("data9055/trainData_to_shop_distance/train_" + mall_id + ".csv", "w")
    file.write("row_id")
    for shop_id in MALL_SHOPS[mall_id]:
        file.write(","+shop_id)
    file.write("\n")
    for index, row in traindata.iterrows():
        file.write(str(row.row_id))
        # file.write(",")
        for shop_id in MALL_SHOPS[mall_id]:
            file.write(","+str(produceLocationInfo(row.trade_latitude, row.trade_longitude, shopid_long_lat[shop_id][1],
                                               shopid_long_lat[shop_id][0])))
        file.write("\n")
    file.close()
#######################################
    print "正在写入"+mall_id+"的测试集"
    testdata = pd.read_csv("data9055/testData/" + mall_id + ".csv")
    file = open("data9055/testData_to_shop_distance/test_" + mall_id + ".csv", "w")
    file.write("row_id")
    for shop_id in MALL_SHOPS[mall_id]:
        file.write(","+shop_id)
    file.write("\n")
    for index, row in testdata.iterrows():
        file.write(str(row.row_id))
        # file.write(",")
        for shop_id in MALL_SHOPS[mall_id]:
            file.write(","+str(produceLocationInfo(row.trade_latitude, row.trade_longitude, shopid_long_lat[shop_id][1],
                                               shopid_long_lat[shop_id][0])))
        file.write("\n")
    file.close()
    print "#################"