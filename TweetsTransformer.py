import re 
import nltk
import string
import contractions
from copy import deepcopy
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
import pandas as pd
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)


arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                            ـ    | # Tatwil/Kashida
                         """, re.VERBOSE)

arabic_pattern = re.compile('[\u0600-\u06FF]')

class TweetsTransformer():
    def __init__(self,usr_rm = True, hsh_rm=True, number_rm = True, URL_rm = True, emoji_rm = True, lower_eff = True, extra_space_rm = True, con_pun = True, non_english_rm = True, Stemming = False, meaningless_rm = True, Lemmatizing = False, arabic_norm = False, new_line_rm = True) -> None:
        self.usr_rm = usr_rm
        self.hsh_rm = hsh_rm
        self.number_rm = number_rm
        self.URL_rm = URL_rm
        self.emoji_rm = emoji_rm
        self.lower_eff = lower_eff
        self.extra_space_rm = extra_space_rm
        self.con_pun = con_pun
        self.non_english_rm = non_english_rm
        self.Stemming = Stemming
        self.meaningless_rm = meaningless_rm
        self.Lemmatizing = Lemmatizing
        self.arabic_norm = arabic_norm
        self.new_line_rm = new_line_rm
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_text = X.apply(self.users_removal)
        cleaned_text = cleaned_text.apply(self.URL_removal)
        cleaned_text = cleaned_text.apply(self.emoji_removal)
        cleaned_text = cleaned_text.apply(self.hashtags_removal)
        cleaned_text = cleaned_text.apply(self.meaningless_removal)
        cleaned_text = cleaned_text.apply(self.numbers_removal)
        cleaned_text = cleaned_text.apply(self.new_line_removal)
        cleaned_text = cleaned_text.apply(self.non_english_removal)
        cleaned_text = cleaned_text.apply(self.non_arabic_removal)
        cleaned_text = cleaned_text.apply(self.lower)
        cleaned_text = cleaned_text.apply(self.contractions_punctuation_fix)
        cleaned_text = cleaned_text.apply(self.extra_space_removal)
        cleaned_text = cleaned_text.apply(self.stemming)
        cleaned_text = cleaned_text.apply(self.lemmatizeing)
        cleaned_text = cleaned_text.apply(self.character_normalization_arabic)
        cleaned_text = cleaned_text.apply(self.diacritics_removal)
        cleaned_text = cleaned_text.apply(self.non_arabic_removal)
        cleaned_text = cleaned_text.apply(self.text_elongation_removal)
        return cleaned_text 
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def numbers_removal(self, text):
        if not self.number_rm:
            return text
        cleaned_text = re.sub(r'\d+', ' ', text)
        return cleaned_text

    def users_removal(self, text):
        if not self.usr_rm:
            return text
        return re.sub(r"@\w+", " ", text)
    
    def lower(self, text):
        if not self.lower_eff:
            return text
        return text.lower()
    
    def emoji_removal(self, text):
        if not self.emoji_rm:
            return text
        return emoji_pattern.sub(r' ', text)
    
    def non_english_removal(self, text):
        if not self.non_english_rm:
            return text
        return re.sub(r"[^a-zA-Z' ]+", ' ', text)
    
    def URL_removal(self, text):
        if not self.URL_rm:
            return text
        return re.sub(r"https?:\/\/.*[\r\n]*", ' ', text)
    
    def contractions_punctuation_fix(self, text):
        if not self.con_pun:
            return text
        cleaned_text = contractions.fix(text)
        cleaned_text = ''.join([c for c in cleaned_text if c not in string.punctuation])
        cleaned_text = re.sub(r'؟', ' ', text) 
        return cleaned_text.strip()
    
    def hashtags_removal(self, text):
        if not self.hsh_rm:
            return text
        return re.sub(r"#\w+", " ", text)

    def extra_space_removal(self, text):
        if not self.extra_space_rm:
            return text
        cleaned_text = re.sub('\ +', ' ', text).strip()
        return cleaned_text
    
    def stemming(self,text):
      if(not self.Stemming):
        return text
      stemmer = PorterStemmer()
      words = nltk.word_tokenize(text)

      # Stem each word using the Porter stemming algorithm
      stemmed_words = [stemmer.stem(word) for word in words]
      return " ".join(stemmed_words)
    
    def lemmatizeing(self,text):

      if (not self.Lemmatizing):
        return text
      lemmatizer = WordNetLemmatizer()
      words = nltk.word_tokenize(text)

      # Stem each word using the Porter stemming algorithm
      lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
      return " ".join(lemmatized_words)

    def meaningless_removal(self, text):
        if (not self.meaningless_rm):
            return text
        return re.sub("\w*[^a-zA-Z ]\w*", ' ', text)
    def character_normalization_arabic(self,text):
        if (not self.arabic_norm):
            return text
        preprocessed_text = re.sub("[إأآا]", "ا", text)
        preprocessed_text = re.sub("ى", "ي", preprocessed_text)
        preprocessed_text = re.sub("ؤ", "ء", preprocessed_text)
        preprocessed_text = re.sub("ئ", "ء", preprocessed_text)
        preprocessed_text = re.sub("ة", "ه", preprocessed_text)
        preprocessed_text = re.sub("گ", "ك", preprocessed_text)
        preprocessed_text = re.sub("ڤ", "ف", preprocessed_text)
        preprocessed_text = re.sub("چ", "ج", preprocessed_text)
        preprocessed_text = re.sub("ژ", "ز", preprocessed_text)
        preprocessed_text = re.sub("پ", "ب", preprocessed_text)
        return preprocessed_text
    def diacritics_removal(self, text):
        if not self.arabic_norm:
            return text
        preprocessed_text = re.sub(arabic_diacritics, '', text)
        return preprocessed_text
    def non_arabic_removal(self, text):
        if not self.arabic_norm:
            return text
        preprocessed_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return preprocessed_text
    def new_line_removal(self, text):
      if not self.new_line_rm:
        return text
      preprocessed_text = text.replace('\n', ' ') 
      return preprocessed_text
    def text_elongation_removal(self, text):
        if not self.arabic_norm:
            return text
        preprocessed_text = re.sub(r'(\w)\1{2,}', r'\1\1', text) 
        return preprocessed_text
