import os
import re
import jieba

labels = (("1", "体育"),
          ("2", "军事"),
          ("3", "娱乐"),
          ("4", "旅行"),
          ("5", "游戏"),
          ("6", "社会"),
          ("7", "科技"),
          ("8", "财经")
          )

stop_words = list()
with open(os.getcwd() + '\\stop_words.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())

words_list = list()
for label in labels:
    for root, dirs, files in os.walk(os.getcwd() + '\\data_set\\' + label[1]):
        count = 1
        for item in files:
            words_line = ""
            with open(os.path.join(root, item), 'r', encoding='utf-8') as f:
                content = f.read().split('\n', 2)[2]
                text = content.replace('\n', ' ')
                words = jieba.cut(text.strip())
                for word in words:
                    word = word.strip()
                    if word not in stop_words:
                        if not re.match('[a-zA-Z0-9\s!]', word):
                            words_line += word + ' '

            print(words_line)
            words_list.append(words_line)
            if count == 100:  # Only use the first 100 articles
                break
            else:
                count += 1

with open('words_set.txt', 'w', encoding='utf-8') as f:
    for item in words_list:
        f.writelines(item + '\n')
