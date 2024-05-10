import os
import jieba
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import code
import matplotlib.pyplot as plt

# 加载训练数据
def load_data(data_path):
    data = pd.read_csv(data_path)
    texts = data["Title"].tolist()
    labels = data["label"].tolist()
    return texts, labels


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
    stopwords = ['是', '的', '了', '在', '和', '有', '被', '这', '那', '之', '更', '与', '对于', '并', '我', '他', '她',
                 '它', '我们', '他们', '她们', '它们']

    processed_texts = []
    for text in texts:
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in stopwords]
        processed_texts.append(' '.join(words))
    return processed_texts


# 加载数据集
data_path = 'train.news.csv' # 数据集路径
texts, labels = load_data(data_path)

# 预处理数据集
tokenized_texts = tokenize(texts)
processed_texts = remove_stopwords_punctuation(tokenized_texts)

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(processed_texts, labels, test_size=0.2,
                                                                    random_state=42)

# 文本向量化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

vocab_size = len(tokenizer.word_index) + 1

max_len = 100  # 设定文本的最大长度
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
val_sequences = pad_sequences(val_sequences, maxlen=max_len, padding='post')

# 构建CNN模型
embedding_dim = 100
num_filters = 128
filter_sizes = [3, 4, 5]

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Conv1D(num_filters, filter_sizes[0], activation='relu'))
model.add(Conv1D(num_filters, filter_sizes[1], activation='relu'))
model.add(Conv1D(num_filters, filter_sizes[2], activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 训练模型
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

model.fit(train_sequences, train_labels, validation_data=(val_sequences, val_labels), epochs=10, batch_size=32)

# 加载测试数据
test_path = "test.feature.csv"  # 测试集的路径
test_texts, test_ids = load_test_data(test_path)

# 预处理测试数据
tokenized_test_texts = tokenize(test_texts)
processed_test_texts = remove_stopwords_punctuation(tokenized_test_texts)
test_sequences = tokenizer.texts_to_sequences(processed_test_texts)
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# 进行预测
y_pred = (model.predict(test_sequences) > 0.5).astype(int)

# 输出测试结果
results_df = pd.DataFrame({'id': test_ids, 'label': y_pred.flatten()})
results_df.to_csv('result_cnn_jieba.csv', index=False)

