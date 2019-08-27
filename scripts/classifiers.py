import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
#from TextProcessor import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import warnings
import matplotlib.pyplot as plt

np.random.seed(2019)
STOPWORDS = set(stopwords.words('spanish'))
nltk.download('wordnet')
# Ignoring warnings
warnings.filterwarnings('ignore')

def result_graph(history, algorithm_name):
	# function to graph the results
	plt.title('Loss')
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.savefig('loss_' + algorithm_name + '.png')
	plt.clf()
	plt.close()
	plt.title('Accuracy')
	plt.plot(history.history['acc'], label='train')
	plt.plot(history.history['val_acc'], label='test')
	plt.legend()
	plt.savefig('acc_' + algorithm_name + '.png')
	plt.clf()
	plt.close()

def save_model_weight(model, algorithm_name):
	# save model and weight
	model.save("model" + algorithm_name + ".h5")
	model_json = model.to_json()
	with open("model" + algorithm_name + ".json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model" + algorithm_name + ".h5")

# Read data
# Training and Test
df_model_training = pd.read_csv('../data/final_ecuador_data/final_training.csv')
df_model_test = pd.read_csv('../data/final_ecuador_data/final_test.csv')
df_model_training.category.value_counts(normalize=False)
df_model_test.category.value_counts(normalize=False)

def process_news(text, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
	d = {'texto': [text]}
	df = pd.DataFrame(data=d)
	corpus = df["texto"]
	vectorizer = TfidfVectorizer()
	matrix = vectorizer.fit_transform(corpus)
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(corpus.values)
	word_index = tokenizer.word_index
	X = tokenizer.texts_to_sequences(corpus.values)
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	return X

# Prediction
# Using the training and test dataset, we will predict the category of news.
# SVM Classifier
frames = [df_model_training, df_model_test]
df_model = pd.concat(frames)
# Create TFIDF matrix.
corpus = df_model['tokens-headline-stopwords-stemming']
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus)
print(matrix.shape)
# Split my data on training y test.
num_training = len(df_model_training)
X_train = matrix[:num_training, :]
X_test = matrix[num_training:, :]
y_train = df_model["category"].values[:num_training]
y_test = df_model['category'].values[num_training:]

print("\n\n\n#################### SVM #####################################################################\n\n\n")
# Create the SVM classifier
clf = SVC(probability=True, kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Any Neural Network Classifier
print("\n\n\n#################### LSTM #####################################################################\n\n\n")
def LSTM_(corpus, df_model):
	MAX_NB_WORDS = 350			# The maximum number of words
	MAX_SEQUENCE_LENGTH = 20	# Max number of words in each complaint.
	EMBEDDING_DIM = 10
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(corpus.values)
	word_index = tokenizer.word_index
	print('%s unique tokens.' % len(word_index))

	X = tokenizer.texts_to_sequences(corpus.values)
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	print('data tensor shape:', X.shape)

	Y = pd.get_dummies(df_model['category'])
	labels = Y.columns
	target = Y.as_matrix()
	Y = target.copy()
	print('Shape of label tensor:', target.shape)

	X_train = X[:num_training,:]
	X_test = X[num_training:,:]
	y_train = Y[:num_training]
	y_test = Y[num_training:]

	model = Sequential(name='ALG_1')
	model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1], name='Embedding'))
	model.add(SpatialDropout1D(0.2, name='SpatialDropout1D'))
	model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2, name='LSTM'))
	model.add(Dense(6, activation='softmax', name='Dense_softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	plot_model(model, to_file='LSTM.png',show_shapes=True, show_layer_names=True)
	print(model.summary())
	epochs = 60
	batch_size = 32
	history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
						validation_split=0.20, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
	result = model.evaluate(X_test,y_test)
	print('Test \n  Loss: {:0.4f} \nAccuracy: {:0.4f}'.format(result[0],result[1]))

	save_model_weight(model, "LSTM")
	result_graph(history, "LSTM")
	# prediction
	news = "La Policía colombiana desarticuló una red de narcotraficantes dedicada a la elaboración y venta de drogas sintéticas que eran distribuidas en Ecuador"
	print("Prediction of news LSTM: " + news)
	ynew = model.predict(process_news(news, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH))
	index_max = np.argmax(ynew[0])
	print("result: " + labels[index_max])
#LSTM_(corpus, df_model)

print("\n\n\n#################### CNN #####################################################################\n\n\n")

def CNN_(corpus, df_model):
	MAX_NB_WORDS = 500			# The maximum number of words
	MAX_SEQUENCE_LENGTH = 15	# Max number of words in each complaint.
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(corpus.values)
	word_index = tokenizer.word_index
	print('%s unique tokens.' % len(word_index))
	X = tokenizer.texts_to_sequences(corpus.values)
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	print('data tensor shape:', X.shape)
	Y = pd.get_dummies(df_model['category'])
	labels = Y.columns
	target = Y.as_matrix()
	Y = target.copy()
	print('Shape of label tensor:', target.shape)

	X_train = X[:num_training,:]
	X_test = X[num_training:,:]
	y_train = Y[:num_training]
	y_test = Y[num_training:]

	print(X_train.shape,y_train.shape)
	print(X_test.shape,y_test.shape)

	model = Sequential()
	model.add(Embedding(input_dim=MAX_NB_WORDS,output_dim=64,input_length=X.shape[1], trainable=True))
	model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
	model.add(layers.MaxPooling1D(8, padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dense(6, activation ='softmax'))
	plot_model(model, to_file='CNN.png',show_shapes=True, show_layer_names=True)

	model.compile(loss='categorical_crossentropy',optimizer='rmsprop',  metrics=['acc'])
	history = model.fit(X_train, y_train, epochs=70, batch_size=10,validation_split=0.3, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
	result = model.evaluate(X_test,y_test)
	print('Test \n  Loss: {:0.4f} \nAccuracy: {:0.4f}'.format(result[0],result[1]))
	save_model_weight(model, "CNN")
	result_graph(history, "CNN")
    
	print(model.summary())
	news = "La Policía colombiana desarticuló una red de narcotraficantes dedicada a la elaboración y venta de drogas sintéticas que eran distribuidas en Ecuador"
	print("Prediction of news CNN: " + news)
	ynew = model.predict(process_news(news, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH))
	index_max = np.argmax(ynew[0])
	print("result: " + labels[index_max])
	

CNN_(corpus, df_model)
