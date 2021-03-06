# -*- coding: utf-8 -*-
"""intent_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hN0HOR4bOQLV8b4igjEpMaUTSlrSHfM8
"""

# !pip install -q -U tensorflow==1.8.0
# !pip install -q -U sklearn
# !pip install -q -U  numpy
# !pip install -q -U spacy
# !python -m spacy download en

import numpy as np
np.random.seed(1)



import numpy as np
import tensorflow as tf
import spacy
nlp = spacy.load('en')

#training data

X =['i want to cancel', 'cancel that', 'cancel', 'im looking for a place in banglore serving Chinese', u"i'm looking for Chinese food", u"I'm looking for south indian places", 'im looking for a place near banglore', u"i'm looking for a place to eat near down town la", u"i'm looking for a place in new york", 'im looking for a place in banglore', 'looking for indian cuisine in new york', 'central indian restaurant', 'I am looking for mexican indian fusion', 'I am looking a restaurant in 29432', 'I am looking for asian fusion food', 'anywhere near 18328', 'anywhere in the west', 'search for restaurants', 'i am looking for an indian spot called olaolaolaolaolaola', 'show me a mexican place in the centre', 'show me chines restaurants in the north', 'show me chinese restaurants', u"i'm looking for a place in the north of town", 'I am searching for a dinner spot', 'I want to grab lunch', u"i'm looking for a place to eat", 'dear sir', 'good evening', 'good morning', 'hi', 'hello', 'hey there', 'howdy', 'hey', 'sounds really good', 'great choice', 'correct', 'right, thank yo', 'great', 'ok', u"that's right", 'indeed', 'yeah', 'yep', 'yes', 'have a good one', 'Bye bye', 'farewell', 'end', 'stop', 'good bye', 'goodbye', 'bye', 'thank you iky', 'thanks', 'thank you very much']
y = ['cancel', 'cancel', 'cancel', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'restaurant_search', 'greet', 'greet', 'greet', 'greet', 'greet', 'greet', 'greet', 'greet', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'affirm', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'thank_yo', 'thank_yo', 'thank_yo']

# spacy context vector size
vocab_size = 384

# create spacy doc vector matrix
x_train = np.array([list(nlp(unicode(x)).vector) for x in X])
print(x_train)
print(x_train.shape)

num_labels = len(set(y))
print(num_labels)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(y)
y_train = encoder.transform(y)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(vocab_size,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
#
# input_shape=(vocab_size,)
# model.add(tf.keras.layers.Reshape(input_shape + (1, ), input_shape=input_shape))
# model.add(tf.keras.layers.Conv1D(64, 2, strides=1, padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPool1D(pool_size=5))
# model.add(tf.keras.layers.Conv1D(128, 2, strides=1, padding='same', activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(num_labels, activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,shuffle=True, epochs=300, verbose=1)

query = u"i agree to that"

# x_predict = tokenize.texts_to_matrix([query])

x_predict = [nlp(query).vector]

prediction = model.predict(np.array([x_predict[0]]))
print(prediction)
text_labels = encoder.classes_ 
predicted_label = text_labels[np.argmax(prediction[0])]
print("Predicted label: " + predicted_label )

print("Confidence: %.2f"% prediction[0].max())

loss, accuracy = model.evaluate(x_train, y_train)
print('Test accuracy: %.2f' % (accuracy))