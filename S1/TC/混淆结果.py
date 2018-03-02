import pandas as pd
import numpy as np
# import xgboost
# import xgboost
# ## 如果预测的结果只是一个mall,则使用下面的代码补全没有预测的记录
# mallid = "m_1292"
# test = pd.read_csv("test.csv")
# results = pd.read_csv('data9055/result/lgb_' + mallid + "_dist_new.csv")
# results = pd.merge(test[['row_id']], results, on='row_id', how='left')
# results.fillna('0',inplace=True)
# results.to_csv("only_lgb_" + mallid +"_dist_new.csv",index=False)

######################## 把该mall里面所有的待预测的清0  ########################
# mallid = "m_4079"
# clear = pd.read_csv('data9055/testData/' + mallid + ".csv")
# clearRowids = set()
# for ix, row in clear.iterrows():
#     clearRowids.add(row['row_id'])
#
# result = pd.read_csv("9098.csv")
# out = open('9098_sub_' + mallid + '.csv', 'w')
# out.write("row_id,shop_id\n")
# for ix, row in result.iterrows():
#     curRowid = row['row_id']
#     if curRowid in clearRowids:
#         out.write(str(curRowid) + "," + "s_0" + "\n")
#     else:
#         out.write(str(curRowid) + "," + row['shop_id'] + "\n")
# out.close()
# Merge single mall file in every algorithm to the result.
# 返回一个set 里面包含所有的mall_id值
DIR_PREFIX = "data9055/"


def getMalls():
    r = set()
    df = pd.read_csv("train_shopinfo.csv")
    malls = df['mall_id'].values
    for mall in malls:
        r.add(mall)
    return r
###########################################

for mallid in getMalls():
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
    ##########################################################################
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
    lgb_data = pd.read_csv(
        'data9055/model_file/model3/' + mallid + '_model3.csv')
    xgb_index_list = []
    rf_index_list = []
    lgb_index_list = []
    for i in range(xgb_data.shape[0]):
        xgb_value_list = np.asarray([float(i)
                                     for i in xgb_data.iloc[i, 1].split(",")])
        rf_value_list = np.asarray([float(i)
                                    for i in rf_data.iloc[i, 1].split(",")])
        lgb_value_list = np.asarray([float(i)
                                     for i in lgb_data.iloc[i, 1].split(",")])
        xgb_max_index = np.argmax(xgb_value_list)
        rf_max_index = np.argmax(rf_value_list)
        lgb_max_index = np.argmax(lgb_value_list)
        xgb_index_list.append(xgb_max_index)
        rf_index_list.append(rf_max_index)
        lgb_index_list.append(lgb_max_index)

    result_xgb = pd.concat([test_rowId, pd.Series(xgb_index_list)], axis=1)
    result_rf = pd.concat([test_rowId, pd.Series(rf_index_list)], axis=1)
    result_lgb = pd.concat([test_rowId, pd.Series(lgb_index_list)], axis=1)
    result_xgb.columns = ['row_id', 'ClassIndex']
    result_rf.columns = ['row_id', 'ClassIndex']
    result_lgb.columns = ['row_id', 'ClassIndex']

    shop_id = []
    for index, row in result_xgb.iterrows():
        shop_id.append(classIndexShopMap[str(row['ClassIndex'])])
    result_xgb['shop_id'] = pd.DataFrame(shop_id)
    del result_xgb['ClassIndex']

    shop_id = []
    for index, row in result_rf.iterrows():
        shop_id.append(classIndexShopMap[str(row['ClassIndex'])])
    result_rf['shop_id'] = pd.DataFrame(shop_id)
    del result_rf['ClassIndex']

    shop_id = []
    for index, row in result_lgb.iterrows():
        shop_id.append(classIndexShopMap[str(row['ClassIndex'])])
    result_lgb['shop_id'] = pd.DataFrame(shop_id)
    del result_lgb['ClassIndex']

    # 写入结果到文件
    result_xgb.to_csv(DIR_PREFIX + "model_file/model1_cat/" +
                      mallid + ".csv", index=False)
    result_rf.to_csv(DIR_PREFIX + "model_file/model2_cat/" +
                     mallid + ".csv", index=False)
    result_lgb.to_csv(DIR_PREFIX + "model_file/model3_cat/" +
                      mallid + ".csv", index=False)

print("result1")
xgb_result = []
for mallid in getMalls():
    mall_data = pd.read_csv(DIR_PREFIX+"model_file/model1_cat/"+mallid+".csv")
    xgb_result.append(mall_data)
result = pd.concat(xgb_result)
result.to_csv("data9055/model_file/result1.csv",index=False)

print("result2")
rf_result = []
for mallid in getMalls():
    mall_data = pd.read_csv(DIR_PREFIX+"model_file/model2_cat/"+mallid+".csv")
    rf_result.append(mall_data)
result = pd.concat(rf_result)
result.to_csv("data9055/model_file/result2.csv",index=False)
print("result3")
lgb_result = []
for mallid in getMalls():
    mall_data = pd.read_csv(DIR_PREFIX+"model_file/model3_cat/"+mallid+".csv")
    lgb_result.append(mall_data)
result = pd.concat(lgb_result)
result.to_csv("data9055/model_file/result3.csv",index=False)
