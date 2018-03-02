import pandas as pd

# 分析店铺各个类别的比例
mallid = 'm_7168'
train = pd.read_csv('data9055/trainDataWithClassIndex/' + mallid +'.csv')
for row in train['ClassIndex'].value_counts():
    print(row)