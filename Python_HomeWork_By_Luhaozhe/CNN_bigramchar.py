import pandas as pd
import jieba
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

# 获取数据


train_data = pd.read_csv('train.news.csv')
test_data = pd.read_csv('test.feature.csv')

print("finish read csv")

# 获取预训练 word2vec 并构建词表
word2vec = open("sgns.sogounews.bigram-char", "r", encoding='UTF-8')
t = word2vec.readline().split()
n, dimension = int(t[0]), int(t[1])
print(n)
print(dimension)
wordAndVec = word2vec.readlines()
wordAndVec = [i.split() for i in wordAndVec]
vectorsMap = []
word2index = {}
index2word = {}
for i in range(n):
    vectorsMap.append(list(map(float, wordAndVec[i][len(wordAndVec[i]) - dimension:])))
    word2index[wordAndVec[i][0]] = i
    index2word[i] = wordAndVec[i][0]

word2vec.close()
print("finish reading")

# jieba 分词与词向量构建
features_train = []
features_test = []
for text in train_data['Title']:
    word_feature = []
    for word in jieba.cut(text):
        if word in word2index:
            word_feature.append(vectorsMap[word2index[word]])
    features_train.append(word_feature)

for text in test_data['Title']:
    word_feature = []
    for word in jieba.cut(text):
        if word in word2index:
            word_feature.append(vectorsMap[word2index[word]])
    features_test.append(word_feature)


print("finish creating features")

# 模型输入构建

max_len1 = max([len(i) for i in features_train])
max_len2 = max([len(i) for i in features_test])
max_len = max(max_len1, max_len2)
X_train = []
X_test = []
for sen in features_train:
    tl = sen
    tl += [[0] * 300] * (max_len - len(tl))
    X_train.append(tl)
for sen in features_test:
    tl = sen
    tl += [[0] * 300] * (max_len - len(tl))
    X_test.append(tl)

print("finish creating X_train X_test")

Y_train = train_data['label']

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)

np.random.seed(1)
np.random.shuffle(X_train)
np.random.seed(1)
np.random.shuffle(Y_train)

split = len(X_train) // 3
X_validation = X_train[:split]
X_train_split = X_train[split:]
Y_validation = Y_train[:split]
Y_train_split = Y_train[split:]

print("finish creating X_train_split, X_validation, Y_split, Y_validation")



# 模型构建
def cnn(X_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(128, 4, activation='relu'),
        tf.keras.layers.MaxPool1D(),
        tf.keras.layers.Convolution1D(64, 5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    print(model.summary())
    return model


print("finish creating model")

# 模型训练
model = cnn(X_train)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(X_train_split, Y_train_split, epochs=10,
                    batch_size=128, verbose=1,
                    validation_data=(X_validation, Y_validation))

model.save('weChatFakeNewsDetection')

predictions = model.predict(X_test)

np.savetxt('predict.csv', predictions, delimiter=',', fmt='%f')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()
