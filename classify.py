import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer

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

def printKFold(classifier, X, y, k=5):
    print("\nWith %d-fold cross validation" % (k))
    kf = KFold(n_splits=k)
    scores = cross_val_score(classifier, X, y, cv=kf)
    print("Accuracy: %0.2f%% (+/- %0.2f%%)\n" % (scores.mean() * 100, scores.std() * 2 * 100))

def main():
	human_data = pd.read_csv('data/human_data.txt', sep='\t')
	# print(human_data.head())
	chimp_data = pd.read_csv('data/chimp_data.txt', sep='\t')
	# print(chimp_data.head())
	dog_data = pd.read_csv('data/dog_data.txt', sep='\t')
	# print(dog_data.head())

	human_data['words'] = human_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	human_data = human_data.drop('sequence', axis=1)
	# print(human_data.head())
	chimp_data['words'] = chimp_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	chimp_data = chimp_data.drop('sequence', axis=1)
	# print(chimp_data.head())
	dog_data['words'] = dog_data.apply(lambda x: Encode_KMer(x['sequence']), axis=1)
	dog_data = dog_data.drop('sequence', axis=1)
	# print(dog_data.head())

	human_texts = list(human_data['words'])
	for item in range(len(human_texts)):
	    human_texts[item] = ' '.join(human_texts[item])
	label_human = human_data.iloc[:, 0].values

	chimp_texts = list(chimp_data['words'])
	for item in range(len(chimp_texts)):
	    chimp_texts[item] = ' '.join(chimp_texts[item])
	label_chimp = chimp_data.iloc[:, 0].values

	dog_texts = list(dog_data['words'])
	for item in range(len(dog_texts)):
	    dog_texts[item] = ' '.join(dog_texts[item])
	label_dog = dog_data.iloc[:, 0].values

	cv = CountVectorizer(ngram_range=(4,4))
	X_human = cv.fit_transform(human_texts)
	print('Learning with human data.')
	X_chimp = cv.transform(chimp_texts)
	X_dog = cv.transform(dog_texts)

	print(X_human.shape)
	print(X_chimp.shape)
	print(X_dog.shape)

	# ax = human_data['class'].value_counts().sort_index().plot.bar(title='Number of gene sequences for each function in human DNA data', color='#E0AC69')
	# for p in ax.patches:
	# 	# ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
	# 	ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

	# plt.xlabel("Gene functions")
	# plt.ylabel("Number of sequences in input data")
	# plt.show()

	# ax = chimp_data['class'].value_counts().sort_index().plot.bar(title='Number of gene sequences for each function in chimp DNA data', color='#674E38')
	# for p in ax.patches:
	# 	# ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
	# 	ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

	# plt.xlabel("Gene functions")
	# plt.ylabel("Number of sequences in input data")
	# plt.show()

	# ax = dog_data['class'].value_counts().sort_index().plot.bar(title='Number of gene sequences for each function in dog DNA data', color='#E0CFAA')
	# for p in ax.patches:
	# 	# ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
	# 	ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

	# plt.xlabel("Gene functions")
	# plt.ylabel("Number of sequences in input data")
	# plt.show()

	# human_data['class'].value_counts().sort_index().plot.bar()
	# chimp_data['class'].value_counts().sort_index().plot.bar()
	# dog_data['class'].value_counts().sort_index().plot.bar()

	X_train, X_test, y_train, y_test = train_test_split(X_human, label_human, test_size = 0.20, random_state=42)
	
	print(X_train.shape)
	print(X_test.shape)
	
	# for a in range(0, 11):
	# classifier = MultinomialNB(alpha=a/10.0)
	# classifier = MultinomialNB(alpha=0.2)
	# classifier = GaussianNB() - dont use
	# classifier = SVC() - dont use
	# classifier = LinearSVC()
	classifier = LogisticRegression(solver='liblinear', multi_class='auto')
	# solver={'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
	classifier.fit(X_train, y_train)

	# human
	y_pred = classifier.predict(X_test)
	print('Testing with human data.')
	# print("Confusion matrix (alpha=%.1f)\n" % (a/10.0))
	print("Confusion matrix\n")
	print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))

	accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
	print("accuracy = %.2f%% \nprecision = %.2f%% \nrecall = %.2f%% \nf1 = %.2f%%" % (accuracy*100, precision*100, recall*100, f1*100))

	printKFold(classifier, X_train, y_train, 5)

	# chimp
	y_pred_chimp = classifier.predict(X_chimp)
	print('Testing with chimp data.')
	# print("Confusion matrix (alpha=%.1f)\n" % (a/10.0))
	print("Confusion matrix\n")
	print(pd.crosstab(pd.Series(label_chimp, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))

	accuracy, precision, recall, f1 = get_metrics(label_chimp, y_pred_chimp)
	print("accuracy = %.2f%% \nprecision = %.2f%% \nrecall = %.2f%% \nf1 = %.2f%%" % (accuracy*100, precision*100, recall*100, f1*100))

	# dog
	y_pred_dog = classifier.predict(X_dog)
	print('Testing with dog data.')
	# print("Confusion matrix (alpha=%.1f)\n" % (a/10.0))
	print("Confusion matrix\n")
	print(pd.crosstab(pd.Series(label_dog, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))

	accuracy, precision, recall, f1 = get_metrics(label_dog, y_pred_dog)
	print("accuracy = %.2f%% \nprecision = %.2f%% \nrecall = %.2f%% \nf1 = %.2f%%" % (accuracy*100, precision*100, recall*100, f1*100))

if __name__ == '__main__':
    main()
