import sys
import os
import re

import numpy as np
import pandas as pd
# from statistics import mean
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def SequenceToArray(seq_str, unknown='x'):
    seq_str = re.sub('[^acgt]', unknown, seq_str.lower())
    return np.array(list(seq_str))

def Encode_Ordinal(seq_array, ordinal_type='float', unknown='x'):
    
    label_encoder = LabelEncoder()
    encoded_seq_array = label_encoder.fit_transform(seq_array)

    if ordinal_type == 'float':
        encoder_map = {'a': 0.25, 'c': 0.50, 'g': 0.75, 't': 1.00, unknown: 0.00}
        encoded_seq_array = []
        for item in seq_array:
            encoded_seq_array.append(encoder_map[item])

    return encoded_seq_array

def Encode_KMer(seq_str, k=6):
    KMers = []
    seq_str = seq_str.lower()
    for i in range(len(seq_str) - k + 1):
        KMers.append(seq_str[i:i+k])

    return KMers

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

def main():
	# pd.set_option('display.max_rows', 10)
	human_data = pd.read_csv('data/human_data.txt', sep='\t')
	# print(human_data.head())
	# chimp_data = pd.read_csv('data/chimp_data.txt', sep='\t')
	# print(chimp_data.head())
	# dog_data = pd.read_csv('data/dog_data.txt', sep='\t')
	# print(dog_data.head())

	# print(human_data.iloc[0]['sequence'])
	# Encode_Ordinal(SequenceToArray(human_data.iloc[0]['sequence']))

	# print(human_data.sequence.map(len).max())
	max_len = human_data.sequence.map(len).max()
	
	# human_data['words'] = human_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	human_data['ordinals'] = human_data.apply(lambda x: Encode_Ordinal(SequenceToArray(x['sequence'])), axis=1)
	human_data['ordinals_padded'] = human_data.apply(lambda x: x['ordinals'] + [0] * (max_len - len(x['ordinals'])), axis=1)
	# human_data['len'] = human_data.apply(lambda x: len(x['ordinals_padded']), axis=1)
	human_data = human_data.drop('sequence', axis=1)
	human_data = human_data.drop('ordinals', axis=1)
	# print(human_data)
	# print(human_data.describe())
	# sys.exit()
	# chimp_data['words'] = chimp_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	# chimp_data = chimp_data.drop('sequence', axis=1)
	# print(chimp_data.head())
	# dog_data['words'] = dog_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	# dog_data = dog_data.drop('sequence', axis=1)
	# print(dog_data.head())

	# human_texts = list(human_data['words'])
	# for item in range(len(human_texts)):
	#     human_texts[item] = ' '.join(human_texts[item])
	label_human = human_data.iloc[:, 0].values

	# chimp_texts = list(chimp_data['words'])
	# for item in range(len(chimp_texts)):
	#     chimp_texts[item] = ' '.join(chimp_texts[item])
	# label_chimp = chimp_data.iloc[:, 0].values

	# dog_texts = list(dog_data['words'])
	# for item in range(len(dog_texts)):
	#     dog_texts[item] = ' '.join(dog_texts[item])
	# label_dog = dog_data.iloc[:, 0].values

	# cv = CountVectorizer(ngram_range=(4,4))
	# X_human = cv.fit_transform(human_texts)
	# X_chimp = cv.transform(chimp_texts)
	# X_dog = cv.transform(dog_texts)

	X_human = pd.DataFrame(human_data['ordinals_padded'].to_list())
	# X_human = human_data.iloc[:, 1:].values
	
	# print(X_human)
	print(X_human.shape)
	# print(X_chimp.shape)
	# print(X_dog.shape)

	# human_data['class'].value_counts().sort_index().plot.bar()
	# chimp_data['class'].value_counts().sort_index().plot.bar()
	# dog_data['class'].value_counts().sort_index().plot.bar()

	X_train, X_test, y_train, y_test = train_test_split(X_human, label_human, test_size = 0.20, random_state=42)
	print(X_train.shape)
	print(X_test.shape)
	
	classifier = KNeighborsClassifier(n_neighbors = 7)

	# classifier = MultinomialNB(alpha=0.1)
	# classifier = GaussianNB() - dont use
	# classifier = SVC() - dont use
	# classifier = LinearSVC()
	# classifier = LogisticRegression(solver='liblinear', multi_class='auto') #solver='lbfgs/liblinear'
	classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_test)

	print("Confusion matrix\n")
	print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

	accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
	print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

if __name__ == '__main__':
    main()
