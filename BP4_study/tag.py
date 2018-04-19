from bs4 import BeautifulSoup

soup = BeautifulSoup('<b class="boldest">Extremely bold</b>', "html.parser")

# tag
tag = soup.b
print(tag)
print(tag.text)
print(tag.string)
print(type(tag))

# tag name
print(tag.name)
tag.name="blockquote"
print(tag)

# tag attributes
print(tag['class'])
print(tag.attrs)
tag['class'] = 'verybold'
tag['id'] = 1
print(tag)
print(tag.attrs)
del tag['class']
del tag['id']
print(tag)
# print(tag['class'])
print(tag.get('class'))


# Multiple-value attributes
css_soup = BeautifulSoup('<p class="body strikeout"></p>', "html.parser")
print(css_soup.p['class'])

css_soup = BeautifulSoup('<p class="body"></p>', "html.parser")
print(css_soup.p['class'])

id_soup = BeautifulSoup('<p id="my id"></p>', "html.parser")
print(id_soup.p['id'])

rel_soup = BeautifulSoup('<p>Back to the <a rel="index">homepage</a></p>', "html.parser")
print(rel_soup.a['rel'])
rel_soup.a['rel'] = ['index', 'contents']
print(rel_soup.p)

xml_soup = BeautifulSoup('<p class="body strikeout"></p>', 'xml')
print(xml_soup.p['class'])
