# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:22:59 2020

@author: dell
"""


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
#print(data)
nltk.download('punkt')
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    
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
    
    
    training = numpy.array(training)
    output = numpy.array(output)
    
    tensorflow.reset_default_graph()
    
    net = tflearn.input_data(shape=[None, len(training[0])]) #none - not gonna specify how many examples we are using but training[0] is the number of input 
    net = tflearn.fully_connected(net, 8) #no. of nodes are 8 in first hidden layer
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # fully connected lyer with output[0] nodes , softmax( gives probabilty for each output class such that it sums up to one for all the classes ... 
    net = tflearn.regression(net)
    
    model = tflearn.DNN(net) #training #we set network
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) #once model is set we give actual inputs and outputs #show metric shows the functionlity of the training which is transparent that how much efficiency how much time , top performances
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)]) #output layer labels k size ki thi to predict krega ki sbse highest probability kis label/tag ki hai training set me se bag of words ki frequency ke refrence leke jiske corresponding output niklega usi training size ka aur uske highest probability ke index use krke output se tag pata krlenge
        results_index = numpy.argmax(results) #highest prob/val after prediction
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        
chat()
