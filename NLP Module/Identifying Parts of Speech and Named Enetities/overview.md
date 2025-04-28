# Text tagging

텍스트의 tag에는 두 종류

## Parts of Speech Tagging(POS)
token을 갖고, 연관된 소리, 태그를 연결

## Named Entity Recognition(NER)
문장
문장이 주어지면 이름으로 된 엔티티들을 뽑아냄
사람, 장소, 조직 등등

# Parts of Speech(POS) tagging

```python
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")
emma_ja = "emma woodhouse "

spacy_doc = nlp(emma_ja)

pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

for token in spacy_doc:
  pos_df = pd.concat([pos_df, pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])], ignore_index=True)

pos_df.head(15)

pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

pos_df_counts.head(10)

pos_df_poscounts = pos_df_counts.groupby(['pos_tag'])['token'].count().sort_values(ascending=False)

pos_df_poscounts.head(10)

nouns = pos_df_counts[pos_df_counts.pos_tag== 'NOUN'][:10]
nouns
```

# Named Entity Recognition (NER)

```python
import spacy
from spacy import displacy
from spacy import tokenizer
import re

nlp = spacy.load('en_core_web_sm')

google_text = ""

spacy_doc = nlp(google_text)

for word in spacy_doc.ents:
  print(word.text, word.label_)

displacy.render(spacy_doc, style="ent") #, jupyter=True)

google_text_clean = re.sub(r'[^\w\s]', '', google_text).lower()
print(google_text_clean)

spacy_doc_clean = nlp(google_text_clean)

for word in spacy_doc_clean.ents:
  print(word.text, word.label_)

displacy.render(spacy_doc_clean, style="ent")


```
