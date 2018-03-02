import os
list = os.listdir('model_output_prob/' )#列出目录下的所有文件和目录
x = set()
for fileName in list:
    x.add(fileName.split('.')[0][:-3])
print(x)
print(len(x))
