#from konlpy.tag import Twitter
from konlpy.tag import Okt
from gensim.models import Word2Vec
import pandas as pd
import csv
"""
@author: lumyjuwon
"""

twitter = Okt()
#twitter = Twitter()

#file = open("../output/train.txt", 'r', encoding='utf-8')
#line = csv.reader(file)
file = open("output/train.txt",'r', encoding='utf-8')
Line=[]

while True:
    tline = file.readline()
    if not tline: break
    Line.append(tline.rstrip('\n'))

token = []
embeddingmodel = []

f = open("output/train_tag.txt",'r',encoding='utf-8')
key=[]

#f2 = open("key.txt","w",encoding='utf-8')

while True:
    nline = f.readline()
    if not nline: break
    key.append(nline.rstrip('\n'))

for i in Line:
    li = i.split('\t')
    content = li[1]  # csv에서 뉴스 제목 또는 뉴스 본문 column으로 변경
#    f2.write(li[0])
#    f2.write("\n")
    sentence = twitter.pos(li[1], norm=True, stem=True)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    embeddingmodel.append(temp_embedding)
    category = li[0]  # csv에서 category column으로 변경
#category_number_dic = {'IT과학': 0, '경제': 1, '정치': 2, '사회': 3, '생활문화': 4}
    category_number_dic = {string:l for l, string in enumerate(key)}
    all_temp.append(category_number_dic.get(category))
    token.append(all_temp)
print("토큰 처리 완료")

"""
for i in line:
    content = i[3]  # csv에서 뉴스 제목 또는 뉴스 본문 column으로 변경
    sentence = twitter.pos(i[3], norm=True, stem=True)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    embeddingmodel.append(temp_embedding)
    category = i[1]  # csv에서 category column으로 변경
    category_number_dic = {'IT과학': 0, '경제': 1, '정치': 2, '사회': 3, '생활문화': 4}
    all_temp.append(category_number_dic.get(category))
    token.append(all_temp)
print("토큰 처리 완료")

"""
embeddingmodel = []
for i in range(len(token)):
    temp_embeddingmodel = []
    for k in range(len(token[i][0])):
        temp_embeddingmodel.append(token[i][0][k])
    embeddingmodel.append(temp_embeddingmodel)
embedding = Word2Vec(embeddingmodel, size=300, window=5, min_count=5, iter=200, sg=1)
embedding.save('post.embedding')
