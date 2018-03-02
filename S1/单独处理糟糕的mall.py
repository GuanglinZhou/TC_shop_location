import pandas as pd

########## 替换糟糕的mall
mallid = 'm_1263'
df = pd.read_csv("m_1263.csv")
mod_rowIdShop = {}
for ix,row in df.iterrows():
    mod_rowIdShop[row['row_id']] = row['shop_id']
mod_rowIds = mod_rowIdShop.keys()

best_df = pd.read_csv('9098.csv')
result = open('9097_modify_' + mallid +'.csv','w')
result.write('row_id,shop_id\n')
for ix,row in best_df.iterrows():
    curRowId = row['row_id']
    if curRowId in mod_rowIds:
        result.write(str(curRowId) + "," + mod_rowIdShop[curRowId] + "\n")
    else:
        result.write(str(curRowId) + "," + row['shop_id'] + "\n")
result.close()