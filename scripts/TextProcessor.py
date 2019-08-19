import sys
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

class TextProcessor(object):

	def __init__(self):
		super(TextProcessor, self).__init__()

	def remove_accents(self,text):
	 	#remove accents in text
	 	if isinstance(text,unicode):
	 		text = unicodedata.normalize('NFKD',text).encode('ASCII', 'ignore')
	 		return text
	 	return text

	def remove_hashtags(self,text):
		regex = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)" # hash-tags
		text = re.sub(regex,' ',text)
		return text

	def remove_mentions(self,text):
		regex = r'@[^\s]+'#remove @-mentions
		text = re.sub(regex,' ',text)
		return text

	def remove_urls(self,text):
		regex = r'http\S+'#remove url
		text = re.sub(regex,' ',text)
		return text

	def remove_punctuation(self,text):
		# r'[.]',#remove points
		regex = r'[%s]' % re.escape(string.punctuation)#signsPattern
		text = re.sub(regex,' ',text)
		return text

	def remove_special_issues(self,text):
		regex_issues = [r'jaja*',r'jeje*',r'[0-5][7-9]*']
		for regex in regex_issues:
			text = re.sub(regex,' ',text)
		return text

	#def remove_stop_words(self,tokens):
	#	stop = stopwords.words('spanish')
	#	stop += stopwords.words('english')
	#	return [token for token in tokens if token not in stop]

	#def normalize_vocals(self,text):
	#	re.sub(r'a*','a',text)
	#	re.sub(r'e*','e',text)
	#	re.sub(r'i*','i',text)
	#	re.sub(r'o*','o',text)
	#	re.sub(r'u*','u',text)
	#	return text

	#def stemming(self,tokens):
	#	return self.stemmer.apply(tokens)

	def tokenization(self,text):
		#this function pre-process de text of a tweet return a list of tokens
		tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
		text = str(text)
		text = text.lower()# text to lower
		text = text.rstrip('\n')
		text = self.remove_accents(text)#remove accents
		text = self.remove_urls(text)#remove urls
		text = self.remove_punctuation(text)#remove sign puntuation
		text = self.remove_special_issues(text)
		#text = self.normalize_vocals(text)
		tokens = tknzr.tokenize(text)

		#tokens = self.remove_stop_words(tokens)#remove stop words

		tokens = [token for token in tokens if len(token)>2]
		text = " ".join(tokens)

		return text
