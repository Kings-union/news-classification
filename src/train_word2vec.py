import multiprocessing
from gensim.models import word2vec

sentences = word2vec.Text8Corpus(r'../data_set/processed_sogou_data_set.txt')
model = word2vec.Word2Vec(sentences, min_count=5, window=5, size=200, workers=multiprocessing.cpu_count())

print(model['季后赛'])
print(model['主场'])
print(model.similarity('季后赛', '文学'))

for i in model.most_similar("男篮"):
    print(i[0], i[1])

print(model.doesnt_match("中超 俱乐部 小组赛 跳投 破门".split()))

model.wv.save_word2vec_format(r'../saved_models/sogou_word2vec.bin', binary=True)
