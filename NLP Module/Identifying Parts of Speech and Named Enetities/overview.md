# Text tagging

텍스트의 tag에는 두 종류

## Parts of Speech Tagging(POS)
token을 갖고, 연관된 소리, 태그를 연결

## Named Entity Recognition(NER)
문장
문장이 주어지면 이름으로 된 엔티티들을 뽑아냄
사람, 장소, 조직 등등

# POS tagging

```python
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")
emma_ja = "emma woodhouse "

spacy_doc = nlp(emma_ja)

```
