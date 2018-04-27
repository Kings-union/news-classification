import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.svm import SVC
import gensim

CATEGORY_NUMBER = 8        # 分类个数
CATEGORY_LENGTH = 100      # 每个分类的数据集大小
MAX_SEQUENCE_LENGTH = 200  # 每条新闻最大长度
EMBEDDING_DIM = 200        # 词向量空间维度
TRAINING_PERCENT = 0.7     # 训练集比例
VALIDATION_PERCENT = 0.1   # 验证集比例
TEST_PERCENT = 0.2         # 测试集比例

# Read data set from file
with open(r'../data_set/words_set.txt', 'r', encoding='utf-8') as f:
    all_texts = [line.strip() for line in f.readlines()]

with open(r'../data_set/labels_set.txt', 'r', encoding='utf-8') as f:
    all_labels = [line.strip() for line in f.readlines()]

# Pre processing data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)  # Update internal vocabulary based on the list of texts
sequences = tokenizer.texts_to_sequences(all_texts)  # Transform each text to a sequence of integers
word_count = len(tokenizer.word_index)  # Get the unique word count in all the texts
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # Pad or truncate the 'text sequence' to the same length
# data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')  # Used TF-IDF in MLP model
labels = to_categorical(np.asarray(all_labels), 8)  # Convert the one-value labels to one-hot vector labels
print('Found %s unique words.' % word_count)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Save words sequence
with open(r'../data_set/words_sequence.txt', 'w', encoding='utf-8') as f:
    f.write(str(tokenizer.word_index))

# Separate training data/label, validation data/label and test data/label
train_data = list()
train_label = list()
val_data = list()
val_label = list()
test_data = list()
test_label = list()
p1 = int(CATEGORY_LENGTH * TRAINING_PERCENT)
p2 = int(CATEGORY_LENGTH * (1 - TEST_PERCENT))
for i in range(CATEGORY_NUMBER):
    part_data = data[(i * 100 + 0):(i * 100 + 100)]
    part_label = labels[(i * 100 + 0):(i * 100 + 100)]
    train_data.extend(part_data[:p1])
    train_label.extend(part_label[:p1])
    val_data.extend(part_data[p1:p2])
    val_label.extend(part_label[p1:p2])
    test_data.extend(part_data[p2:])
    test_label.extend(part_label[p2:])

# Transform to numpy array format
train_data_array = np.array(train_data)
train_label_array = np.array(train_label)
val_data_array = np.array(val_data)
val_label_array = np.array(val_label)
test_data_array = np.array(test_data)
test_label_array = np.array(test_label)
print('train docs: ' + str(len(train_data)))
print('val docs: ' + str(len(val_data)))
print('test docs: ' + str(len(test_data)))

# Build a model
model = Sequential()  # Initialize a sequential model

# word2vec
# w2v_model = gensim.models.KeyedVectors.load_word2vec_format(r'../saved_models/word2vec.bin', binary=True)
# embedding_matrix = np.zeros((word_count+1, EMBEDDING_DIM))
# for word, i in tokenizer.word_index.items():
#     if word in w2v_model:
#         embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
#
# embedding_layer = Embedding(word_count+1, EMBEDDING_DIM, weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH, trainable=False)
# model.add(embedding_layer)

# CNN
model.add(Embedding(input_dim=word_count+1, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))

# LSTM
# model.add(Embedding(input_dim=word_count+1, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
# model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dropout(0.2))

# MLP
# model.add(Dense(512, input_shape=(word_count+1,), activation='relu'))
# model.add(Dropout(0.2))

# Full connection layer
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

# Compile and train the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(train_data_array, train_label_array, validation_data=(val_data_array, val_label_array), epochs=40, batch_size=128)

# Evaluate model
print(model.evaluate(test_data_array, test_label_array))

# Save trained model
model.save(r'../saved_models/cnn_model.h5')
