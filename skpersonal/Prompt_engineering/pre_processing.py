import nltk
import spacy
import pandas as pd
import re


# full_df = pd.read_csv("twcs.csv")
# df = full_df[["author_id","text"]]
# df = df[df['author_id'].str.contains('[a-zA-Z]')]
# print(df.head())
# print(df['author_id'].unique())
# df.to_csv('twcs_authorid_with_text.csv',index = False)
#
df = pd.read_csv("twcs_authorid_with_text.csv")
print(df.head())

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def get_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text)
    if urls:
        return urls
    else:
        return None

df['text_without_urls'] = df['text'].apply(remove_urls)
df['urls'] = df['text'].apply(get_urls)
df = df.drop(['text'],axis = 1)
df.to_csv('twcs_processed.csv',index=False)
print(df.columns)