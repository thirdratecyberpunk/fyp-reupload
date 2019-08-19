'''
Generates text based on the MongoDB dataset.
'''
# imports keras library
import keras
from keras import layers
from keras import models
from keras.preprocessing.text import Tokenizer
# library for accessing MongoDB documents
import pymongo
# libray for printing records in detail
import pprint as pp
import numpy
from bson.objectid import ObjectId

# setting print options for numpy
numpy.set_printoptions(threshold=numpy.nan)

# connecting to the mongo database
myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["text-dataset"]
training = mydb["sample-training"]
testing = mydb["sample-testing"]

# loads all documents from MongoDB
training_documents = training.find()
testing_documents = testing.find()

all_tokens = []

pp.pprint(training_documents[0])

# function to take a training record from MongoDB and turn it into a numpy array
def DictionaryToNumpyArray(mongoRecord):
    result = None
    # gets the columns from the mongo record
    columnNames = map(str, mongoRecord.keys())
    columns = [(str(c), type(mongoRecord[c])) for c in columnNames]

    numpyColumns = []
    # defining columns for numpy data record
    authorCol = numpy.dtype(numpy.string_, columns[0])
    line = numpy.dtype(numpy.string_, columns[1])
    pos = numpy.array(columns[2], dtype=object)
    tags = numpy.array(columns[3], dtype=object)
    tokens = numpy.array(columns[4], dtype=object)

    numpyColumns.append(authorCol)
    numpyColumns.append(line)
    numpyColumns.append(pos)
    numpyColumns.append(tags)
    numpyColumns.append(tokens)
    return numpy.array(numpyColumns)

# def sentenceTo2dVector(sentence):
#     sentence_vec = None
#     for word in sentence:
#         word_vec = np.expand_dims(model[word], axis=0)
#         if sentence_vec is None:
#             sentence_vec = word_vec
#         else:
#             sentence_vec = np.concatenate((sentence_vec, word_vec), axis=0)

x = DictionaryToNumpyArray(training_documents[0])
print (x)

# TODO: transform a record into a tensor

# TODO: transform a tensor into a sentence

# get tensor representations of sentences

# tensor representation of combination

# combine three sentences into a tensor

# TODO: build neural network which takes three sentences as input
# example uses VGG19 neural network

# compute neural style loss

# one loss function should capture difference in terms of words
# maintains "content"

# one loss function should capture difference in terms of word TYPES

# one loss function should capture difference in terms of word dependencies

# combine functions into a single scalar

# calculates gradients of the generated vector
