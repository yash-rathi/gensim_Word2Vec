import pandas as pd
import os
import gensim
from gensim import corpora, models, similarities
import nltk
from nltk.tokenize import word_tokenize,  sent_tokenize
from gensim.models import Word2Vec

#os.chdir("Downloads/")
fd = pd.read_csv("jokes.csv")
print(type(fd))

ques = fd['Question'].values.tolist()
ans = fd['Answer'].values.tolist()
corpus = ques + ans
#print(type(corpus))
#print(corpus)
sents=sent_tokenize("".join(corpus))
corp_tok = [nltk.word_tokenize(se) for se in sents]
#corp_tok = [nltk.word_tokenize(qa.decode('utf-8')) for qa in corpus]
model = Word2Vec(corp_tok, min_count = 1, size = 32, workers=8)

#word = raw_input("Enter any word:\t")
model.most_similar("hi")
