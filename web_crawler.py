import urllib.request
import urllib.error
from bs4 import BeautifulSoup


def get_html_text(url):
    try:
        r = urllib.request.urlopen(url, timeout=15)
        return r.read()
    except:
        return


def get_news_link(url):
    html = get_html_text(url)
    news_dict = dict()
    soup = BeautifulSoup(html, "html.parser")
    for news in soup.find_all(class_='arcticle-list'):
        for title in news.find_all('a'):
            news_dict[title['href']] = title.string
    return news_dict


def get_news_content(url):
    html = get_html_text(url)
    content = ""
    soup = BeautifulSoup(html, "html.parser")
    for news in soup.select("div.article > p"):
        content += news.string + "\n"
    return content

news_dict = get_news_link("http://mil.news.sina.com.cn/")
for link, title in news_dict.items():
    print(link + ' ' + title)
# get_news_content("http://mil.news.sina.com.cn/china/2018-04-19/doc-ifzihneq0427091.shtml")