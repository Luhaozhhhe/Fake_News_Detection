import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
data=pd.read_csv("train.news.csv")
data_validate=pd.read_csv("test.feature.csv")
x,y,x_validate=data['Title'],data['label'],data_validate['Title']
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(x)
x_validate=vectorizer.transform(x_validate)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state = 0)
x_validate1,x_validate2=train_test_split(x_validate,test_size=0.5,random_state=0)
print(type(x_train),type(y_train))
# 朴素贝叶斯
model2 = MultinomialNB()
# K近邻
model3 = KNeighborsClassifier(n_neighbors=50)
# 决策树
model4 = DecisionTreeClassifier(random_state=77)
# 随机森林
model5 = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=10)
# 梯度上升
model6 = GradientBoostingClassifier(random_state=123)
# 支持向量机
model7 = SVC(kernel="rbf", random_state=77)
# 神经网络
model8 = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=77, max_iter=10000)
model_list = [ model2, model3, model4, model5, model6, model7, model8]
model_name = [ '朴素贝叶斯', 'K近邻', '决策树', '随机森林', '梯度上升', '支持向量机', '神经网络']
scores=[]
#采用决策树算法
i=4
model_C = model_list[i]
name = model_name[i]
model_C.fit(x_train, y_train)
s = model_C.score(x_test, y_test)
scores.append(s)
pre = model_C.predict(x_validate)
print(f'{name}方法在测试集的准确率为{round(s, 3)}')