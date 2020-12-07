#CAP5610 DNA CLASSIFICATION

run=60
epochs=10
batch_size=64
v_b=batch_size

from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
plt.ion()


human = pd.read_table('data/human_data.txt')
dog = pd.read_table('data/dog_data.txt')
chimp = pd.read_table('data/chimp_data.txt')


def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
human = human.drop('sequence', axis=1)
chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp = chimp.drop('sequence', axis=1)
dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)
dog = dog.drop('sequence', axis=1)

human_texts = list(human['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])

y_h = human.iloc[:, 0].values                         #y_h for human

chimp_texts = list(chimp['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])

y_c = chimp.iloc[:, 0].values                       # y_c for chimp


dog_texts = list(dog['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])

y_d = dog.iloc[:, 0].values                         # y_d for dog

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length=10000
#max_length = max([len(s.split()) for s in human_texts])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(human_texts)
encoded_docs = tokenizer.texts_to_sequences(human_texts)
X = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post', truncating="post")
vocab_size = len(tokenizer.word_index) + 1

enc_dog = tokenizer.texts_to_sequences(dog_texts)
enc_chimp = tokenizer.texts_to_sequences(chimp_texts)
X_d = pad_sequences(enc_dog, maxlen = max_length, padding = 'post', truncating="post")
X_c = pad_sequences(enc_chimp, maxlen = max_length, padding = 'post', truncating="post")



skf = StratifiedKFold(n_splits = 5, random_state = 14, shuffle = True) 

final=[]

foldnum = 1
for train, test in skf.split(X, y_h):
    # Define the model architecture
    Y = np.array(tf.keras.utils.to_categorical(y_h[train],7, 'int32'))
    Y_t = np.array(tf.keras.utils.to_categorical(y_h[test],7, 'int32'))
    Y_d = np.array(tf.keras.utils.to_categorical(y_d,7, 'int32'))
    Y_c = np.array(tf.keras.utils.to_categorical(y_c,7, 'int32'))
    model = Sequential(name='Bidirectional_LSTM')
    model.add(Embedding(vocab_size, 200, input_length=max_length, name='Embedding1'))
    model.add(LSTM(32, return_sequences=True, name='LSTM1'))
    model.add(Dropout(0.5, name='Dropout1'))
    model.add(Flatten(name='Flatten1'))
    model.add(Dense(50, name='Dense1'))
    model.add(Dense(7, activation='softmax', name = 'Dense2'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy','mae'])
    checkpoint=ModelCheckpoint(f"./run{run}_fold{foldnum}_weights.best.hdf5", monitor='val_accuracy',verbose = 1,
                          save_best_only = True, mode = 'auto')
    print('---------------------')
    print(f"Training fold {foldnum} ...")

    # plot model
    plot_model(model,to_file=f"./model{run}.png", show_shapes=True)
    # Fit data to model
    #history = model.fit(X[train], y_h[train], batch_size=64, epochs=2, verbose=1, callbacks = [checkpoint])
    history = model.fit(X[train], Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X[test],Y_t), validation_batch_size=v_b, callbacks= [checkpoint])

    scores = model.evaluate(X[test], Y_t, verbose=0)
    predicts_c = model.evaluate(X_c,Y_c,verbose=0)
    predicts_d = model.evaluate(X_d,Y_d,verbose=0)
    final.append(scores)
    final.append(predicts_c)
    final.append(predicts_d)
    plt.figure(figsize=(20,15))
    plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20); plt.xlabel('Epoch', fontsize = 20)
    plt.legend(['Train', 'Validation'], fontsize = 20)
    plt.savefig(f"./loss_fig_run{run}_fold{foldnum}.png")

    plt.figure(figsize=(20,15))
    plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20); plt.xlabel('Epoch', fontsize = 20)
    plt.legend(['Train', 'Validation'], fontsize = 20)
    plt.savefig(f"./acc_fig_run{run}_fold{foldnum}.png")
    
    # Increase fold number
    foldnum = foldnum + 1

    
n=1
file = open(f"run{run}_performance.txt", "w") 
for i in final:
    file.write("%s\n" % i)
file.close() 


#Resources
#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
#https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

#You can use plot_model from keras.utils to plot a keras model. link: 
#https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

#https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml-part-2

