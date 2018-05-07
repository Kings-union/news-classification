import re
import jieba

count = 0
stop_words = list()
with open(r'../data_set/stop_words.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        stop_words.append(line.strip())

with open(r'C:\Users\cquan\Desktop\news_sohusite_xml\news_sohusite_xml.dat', 'r', encoding='gbk', errors='ignore') as fs:
    with open(r'../data_set/processed_sogou_data_set.txt', 'w', encoding='utf-8') as fd:
        for line in fs.readlines():
            if line.startswith('<content>'):
                text = re.sub('<content>|</content>', '', line)
                content = text.strip()
                words_line = ""
                if content:  # Filter empty lines
                    words = jieba.cut(content)
                    for word in words:
                        word = word.strip()
                        if word not in stop_words:
                            words_line += word + ' '
                if words_line:
                    fd.write(words_line + '\n')

            count += 1
            if count % 10000 == 0:
                print(count)
