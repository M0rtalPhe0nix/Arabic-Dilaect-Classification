from TweetsTransformer import TweetsTransformer
import pandas as pd
import re
#importing Data
data = pd.read_csv('data.csv', sep=',', encoding='utf-8-sig', engine='python')


preprocessor = TweetsTransformer(lower_eff=False,non_english_rm=False, meaningless_rm=False, arabic_norm=True)
cleaned_data = data[['dialect']].copy(deep = True)
cleaned_data['text'] = preprocessor.fit_transform(data['text'])
cleaned_data.to_csv('cleaned_data.csv', index=False,sep=',', encoding='utf-8-sig', lineterminator='\r\n')
