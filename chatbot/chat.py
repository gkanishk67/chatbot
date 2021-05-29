#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# In[2]:


import numpy
import tflearn
import tensorflow
import random
import json


# In[3]:


import pickle


# In[4]:



with open('intents.json') as file:
    data = json.load(file)


# In[5]:


print(data)


# In[6]:


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output,doc_x,doc_y = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []


# In[7]:



import nltk
nltk.download('punkt')


# In[8]:


for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])


# In[9]:


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)


# In[10]:


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


# In[11]:


training = numpy.array(training)
output = numpy.array(output)


# In[12]:


with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


# In[13]:


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# In[14]:


print(net.shape)


# In[ ]:


# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
 #model.save("model.tflearn")


# In[15]:


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


# In[19]:


def chat():
    print("Hi , Team Meritroad is here for your help (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


# In[19]:

chat()




