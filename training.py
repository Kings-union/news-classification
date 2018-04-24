import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

CATEGORY_NUMBER = 8 # 分类个数
CATEGORY_LENGTH = 100 # 每个分类的数据集大小
MAX_SEQUENCE_LENGTH = 200 # 每条新闻最大长度
EMBEDDING_DIM = 150 # 词向量空间维度
TRAINING_PERCENT = 0.7 # 训练集比例
VALIDATION_PERCENT = 0.1 # 验证集比例
TEST_PERCENT = 0.2 # 测试集比例

# Read data set from file
with open('words_set.txt', 'r', encoding='utf-8') as f:
    all_texts = [line.strip() for line in f.readlines()]

with open('labels_set.txt', 'r', encoding='utf-8') as f:
    all_labels = [line.strip() for line in f.readlines()]

# Pre processing data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

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
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

# Compile and train the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(train_data_array, train_label_array, validation_data=(val_data_array, val_label_array), epochs=40, batch_size=128)

# Save trained model
model.save('word_vector_cnn.h5')

# Evaluate model
print(model.evaluate(test_data_array, test_label_array))
