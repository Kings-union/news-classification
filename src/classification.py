import re
import jieba
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences

labels = {0: "体育", 1: "军事", 2: "娱乐", 3: "旅行", 4: "游戏", 5: "社会", 6: "科技", 7: "财经"}
MAX_SEQUENCE_LENGTH = 200  # 每条新闻最大长度
EMBEDDING_DIM = 200        # 词向量空间维度

# Get stop words
stop_words = list()
with open(r'../data_set/stop_words.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())

# Get words sequence
with open(r'../data_set/words_sequence.txt', 'r', encoding='utf-8') as f:
    words_seq = eval(f.read())

# Get unspecified text
with open(r'../data_set/娱乐/120.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Transform the text to words
words = list()
content = text.replace('\n', ' ')
raw_words = jieba.cut(content.strip())
for word in raw_words:
    word = word.strip()
    if word not in stop_words:
        if not re.match('[a-zA-Z0-9\s!]', word):
            words.append(word)

# Transform the words to sequence
seq_data = [[]]
for word in words:
    if word in words_seq:
        seq_data[0].append(words_seq[word])

data = pad_sequences(seq_data, maxlen=MAX_SEQUENCE_LENGTH)

# Load trained model
m = keras.models.load_model(r'../saved_models/sogou_w2v_model.h5')

# Predict the category of the text
prob = m.predict_proba(np.array(data))[0]
prob_dict = dict()
for i in range(len(prob)):
    prob_dict[labels[i]] = prob[i]

print("The probable classification of input text is: " + labels[m.predict_classes(np.array(data))[0]])
print("The probability ranking of input text is:")
for item in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
    print(item[0] + ': ' + str(format(item[1], '.1%')))
