from konlpy.tag import Twitter
from gensim.models import Word2Vec
import csv

"""
@author: lumyjuwon
"""

twitter = Twitter()

file = open("/home/oslab/nlu/output/Article_shuffled.csv", 'r', encoding='utf-8')
line = csv.reader(file)
token = []
embeddingmodel = []

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


embeddingmodel = []
for i in range(len(token)):
    temp_embeddingmodel = []
    for k in range(len(token[i][0])):
        temp_embeddingmodel.append(token[i][0][k])
    embeddingmodel.append(temp_embeddingmodel)
embedding = Word2Vec(embeddingmodel, size=300, window=5, min_count=10, iter=5, sg=1, max_vocab_size=360000000)
embedding.save('post.embedding_bak')
