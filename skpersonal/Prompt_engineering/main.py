import os
import json
import numpy as np
import pandas as pd
import re
# import nltk
# import spacy
import string
pd.options.mode.chained_assignment = None
from bardapi import Bard

# full_df = pd.read_csv("twcs.csv")
# df = full_df[["text"]]
# df["text"] = df["text"].astype(str)
# full_df.head()

with open('twcs_processed.json') as f:
    data = json.load(f)

record = data[0]

txt = f"Here I give you json file. It contains 'name of the company', 'answer', 'hyperlink' if possible.\n{record}\nAs you given json file, you are now a counseler of the company. Improve 'answer' by using hyper link. If you can, search details in hyperlink. And show me previous answer and your answer so I can compare both."

os.environ['_BARD_API_KEY'] = 'Ygh70gg15b8tj7rjv3Yl79TmrWF2xhdvBFKY41Mu7_MfcOeDamDlHKcgH7DPSb2yWTnmUQ.'

print(Bard().get_answer(f'{txt}')['content'])