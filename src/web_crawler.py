# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import os
import re


def get_html_text(url):
    try:
        r = urllib.request.urlopen(url, timeout=15)
        return r.read()
    except:
        return


def get_news_link(abbr, url):
    html = get_html_text(url)
    news_dict = dict()
    soup = BeautifulSoup(html, "html.parser")
    reg = "http:\/\/" + abbr + ".*shtml"
    for link in soup.find_all("a", href=re.compile(reg)):
        # print(link['href'] + ' ' + link.text)
        news_dict[link['href']] = link.text
    return news_dict


def get_news_content(url):
    html = get_html_text(url)
    content = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for para in soup.select("div.article > p"):
            if para:
                content += para.text + '\n'
    except TypeError as err:
        print(url)
        print("TypeError caught but will be ignored!\n", err)
    return content


def store_news(category, abbr, url):
    news = get_news_link(abbr, url)
    # news = get_news_link_from_file(category + '.txt')
    sequence = 1
    os.mkdir(r'../data_set/' + category)
    try:
        for link, title in news.items():
            print(link + ' ' + title)
            content = get_news_content(link)
            filename = r'../data_set/' + category + '/' + str(sequence) + '.txt'
            with open(filename, 'w', encoding='utf8') as f:
                f.write(link + '\n')
                f.write(category + '\n')
                f.write(title + '\n')
                f.write(content)
            sequence += 1
    except TypeError as err:
        print("TypeError caught but will be ignored")


def get_news_link_from_file(filename):
    news_dict = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            link = line.strip().split(' ', 1)
            news_dict[link[0]] = link[1]
    return news_dict


if __name__ == "__main__":
    news_list = (("军事", "mil", "http://mil.news.sina.com.cn/"),
                 ("体育", "sports", "http://sports.sina.com.cn/"),
                 ("财经", "finance", "http://finance.sina.com.cn/"),
                 ("娱乐", "ent", "http://ent.sina.com.cn/"),
                 ("科技", "tech", "http://tech.sina.com.cn/"),
                 ("旅行", "travel", "http://travel.sina.com.cn/"),
                 ("社会", "news", "http://news.sina.com.cn/society/"),
                 ("游戏", "games", "http://games.sina.com.cn/"))
    # 财经，旅行，社会，游戏四个的链接比较乱，用的另外的手段
    # 具体来说就是先用get_news_link得到可能的链接保存到文件中，人工去除不对的链接
    # 读取文件，逐个获取链接内容
    # 另外游戏的是正则表达式是div.text > div > p
    i = 6
    store_news(news_list[i][0], news_list[i][1], news_list[i][2])
