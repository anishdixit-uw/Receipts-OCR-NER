from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('/Users/anishdixit/Desktop/Course Documents/SPRING 2023/CSE 599/stanford-ner-2020-11-17/classifiers/english.muc.7class.distsim.crf.ser.gz',
					   '/Users/anishdixit/Desktop/Course Documents/SPRING 2023/CSE 599/stanford-ner-2020-11-17/stanford-ner.jar',
					   encoding='utf-8')

with open("recognized.txt") as file:
    text = file.read()
#text = 'TacoBell Nachos 20.2$ Chips 50$ Total 70$'

tokenized_text = word_tokenize(text)
classified_text = st.tag(tokenized_text)

print(classified_text)