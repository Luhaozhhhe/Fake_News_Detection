#此程序用于实验报告中的一些图表的描绘
#1.七种机器学习方法的ACC绘图
import numpy as np
import matplotlib.pyplot as plt

A=['bayes','k-NN','DTree','RForest','GBoosting','SVC','MLP']
B=[0.951,0.837,0.96,0.961,0.84,0.955,0.959]
plt.bar(A,B,color='skyblue')
plt.title("seven machine learning's accuracy")
plt.xlabel(A)
plt.ylabel(B)
plt.ylim(0.6,1)
plt.show()

#2.所有的模型的AUC比较
import numpy as np
import matplotlib.pyplot as plt

A=['bayes','bayes+jieba','CNN+jieba','CNN+gram','CNN+xmnlp','bert']
B=[0.6137,0.6735,0.7323,0.7627,0.8237,0.8676]
plt.bar(A,B,color='yellow')

plt.title("all models learning's AUC")
plt.xlabel(A)
plt.ylabel(B)
plt.ylim(0.5,1)
plt.show()