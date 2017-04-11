import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import time
import os
import json
import datetime

def generate_input_data():
    training_data = []
    training_data.append({"class":"greeting", "sentence":"how are you?"})
    training_data.append({"class":"greeting", "sentence":"how is your day?"})
    training_data.append({"class":"greeting", "sentence":"good day"})
    training_data.append({"class":"greeting", "sentence":"how is it going today?"})

    training_data.append({"class":"goodbye", "sentence":"have a nice day"})
    training_data.append({"class":"goodbye", "sentence":"see you later"})
    training_data.append({"class":"goodbye", "sentence":"have a nice day"})
    training_data.append({"class":"goodbye", "sentence":"talk to you soon"})

    training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
    training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
    training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
    training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})

    training_data.append({"class": "abuse", "sentence": "Fuck off"})
    training_data.append({"class": "abuse", "sentence": "You piece of shit"})
    training_data.append({"class": "abuse", "sentence": "You are dumb"})
    training_data.append({"class": "abuse", "sentence": "You are worthless"})

    return training_data

def generate_words_and_classes():
    training_data = generate_input_data()
    words = []
    classes = []
    documents = []
    for pattern in training_data:
        w = nltk.word_tokenize(pattern.get('sentence'))
        words.extend(w) # Adding the words generated to the list of words
        documents.append((w, pattern.get('class'))) # Append a tuple containing the tokenized words and the class
        if pattern.get('class') not in classes:
            classes.append(pattern.get('class'))
    return words, classes, documents

def generate_word_stems(words):
    ignore_words = ['?']
    stemmer = LancasterStemmer()
    stems = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    stems = list(set(stems)) # Shortcut to filter out duplicates in a list
    return stems

words, classes, documents = generate_words_and_classes()
stems = generate_word_stems(words)
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(stems), "unique stemmed words", stems)
training = []
output = []
output_empty = [0] * len(classes)

stemmer = LancasterStemmer()
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in stems:
        bag.append(1) if w in pattern_words else bag.append(0)
    training.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1 # doc[1] refers to the class as `doc` is a tuple
    output.append(output_row)

# i = 12

# w = documents[i][0]
# print ([stemmer.stem(word.lower()) for word in w])
# print (training[i])
# print (output[i])

def sigmoid(x):
    return (1.0/(1+np.exp(-x)))

def sigmoid_derivative(x):
    return x*(1.0 - x)

def clean_up_sentence(sentence):
    stemmer = LancasterStemmer()
    words = [nltk.word_tokenize(sentence)][0]
    stems = [stemmer.stem(word) for word in words]
    return stems

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, stems, show_details=False):
    sentence_stems = clean_up_sentence(sentence)
    bag = [0] * len(stems)
    for s in sentence_stems:
        for i,w in enumerate(stems):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: ", w)
    return np.array(bag)

def think(sentence, show_details=False):
    bow = bag_of_words(sentence.lower(), stems, show_details)
    if show_details:
        print ("Sentence {} \n B.O.W {}".format(str(sentence), bow))
    # Input layer is our bag of words
    input_layer = bow
    # Matrix multiplication of input and hidden layer
    hidden_layer = sigmoid(np.dot(input_layer, synapse_0))
    # Final output layer
    output_layer = sigmoid(np.dot(hidden_layer, synapse_1))
    return output_layer

# Function to create the synapses

def train(X, y, hidden_neurons=10, alpha=1, epochs=5000, dropout=False, dropout_percentage=0.5):
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)
    last_mean_error = 1
    # Randomly initialize weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)): #iter(range()) used in the tutorial
        # Feed forward network through input, hidden and output layers
        layer_input = X
        layer_hidden = sigmoid(np.dot(layer_input, synapse_0))
        if dropout:
            layer_hidden *= np.randon.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percentage)[0] * (1.0/(1 - dropout_percentage))

        layer_output = sigmoid(np.dot(layer_hidden, synapse_1))

        layer_output_error = y - layer_output # Calculates how much we missed the target class by

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_output_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_output_error))) )
                last_mean_error = np.mean(np.abs(layer_output_error))
            else:
                print ("ERROR BITCH:", np.mean(np.abs(layer_output_error)), ">", last_mean_error )
                break
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_output_delta = layer_output_error * sigmoid_derivative(layer_output)

        # how much did each layer_hidden value contribute to the layer_output error (according to the weights)?
        layer_hidden_error = layer_output_delta.dot(synapse_1.T)

        # in what direction is the target layer_hidden?
        # were we really sure? if so, don't change too much.
        layer_hidden_delta = layer_hidden_error * sigmoid_derivative(layer_hidden)

        synapse_1_weight_update = (layer_hidden.T.dot(layer_output_delta))
        synapse_0_weight_update = (layer_input.T.dot(layer_hidden_delta))

        if (j>0):
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))

        # Perform the weight update
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_1_weight_update = synapse_1_weight_update
        prev_synapse_0_weight_update = synapse_0_weight_update

    now = datetime.datetime.now()

    # Persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = 'synapses.json'
    with open(synapse_file, 'w') as outfile:
        outfile.write(json.dumps(synapse))
    print("Synapses saved.")

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X,y,hidden_neurons=20,alpha=0.1,epochs=100000,dropout=False, dropout_percentage=0.2)

elapsed_time = time.time() - start_time
print("Processing time ", elapsed_time)


ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

classify("sudo make me a sandwich")
classify("how are you today?")
classify("talk to you tomorrow")
classify("who are you?")
classify("make me some lunch")
classify("how was your lunch today?")
classify("I'm sick of yo shit", show_details=True)
