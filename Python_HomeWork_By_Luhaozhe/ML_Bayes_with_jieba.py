import os
import jieba
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载训练数据
def load_data(data_path):
    data = pd.read_csv(data_path)
    texts = data["Title"].tolist()
    names = data["Ofiicial Account Name"].tolist()
    combined_texts = [str(x) + str(y) for x, y in zip(texts, names)]
    labels = data["label"].tolist()
    return combined_texts, labels

# 加载测试数据
def load_test_data(data_path):
    data = pd.read_csv(data_path)
    texts = data["Title"].tolist()
    ids = data["id"].tolist()
    return texts, ids

# 分词
def tokenize(texts):
    tokenized_texts = []
    for text in texts:
        words = jieba.cut(text)
        tokenized_text = ' '.join(words)
        tokenized_texts.append(tokenized_text)
    return tokenized_texts

# 去除停用词和标点符号
def remove_stopwords_punctuation(texts):
    stopwords = ['是', '的', '了', '在', '和', '有', '更', '与', '对于', '并', '我', '他', '她', '它', '我们', '他们', '她们', '它们']
    processed_texts = []
    for text in texts:
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stopwords]
        processed_texts.append(' '.join(words))
    return processed_texts

# 加载数据集
data_path = 'train.news.csv' and 'test.news.csv'
texts, labels = load_data(data_path)


# 预处理数据集
tokenized_texts = tokenize(texts)
processed_texts = remove_stopwords_punctuation(tokenized_texts)  #进行停用词的查找与去除

# 训练集特征提取
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(processed_texts).toarray()

# 保存特征提取器的配置
feature_names = vectorizer.get_feature_names_out()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# 训练模型
model = MultinomialNB()
model.fit(features, labels)

# 加载测试数据
test_path = "test.feature.csv"  # 数据集的路径
test_texts, test_ids= load_test_data(test_path)

# 分词和去除停用词、标点符号
tokenized_test_texts = tokenize(test_texts)
processed_test_texts = remove_stopwords_punctuation(tokenized_test_texts)

# 在测试集上初始化特征提取器
test_vectorizer = TfidfVectorizer(vocabulary=feature_names)

# 测试集特征提取
test_features = test_vectorizer.fit_transform(processed_test_texts).toarray()

# 加载测试集数据并进行预测

y_pred = model.predict(test_features)

# 输出测试结果
results_df = pd.DataFrame({'id': test_ids, 'label': y_pred})
results_df.to_csv('result.csv', index=False)  #输出result.csv测试点集p