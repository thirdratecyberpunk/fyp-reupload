#!/usr/bin/python
'''
Sequence to sequence example in Keras (word-level).
'''
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import spacy
import argparse
import sys
import string
import TextInformation as textinformation
import random

# defining command line arguments
parser = argparse.ArgumentParser(description='word level sequence to sequence model trained on ENGLISH-FRENCH model.')
parser.add_argument('data_path', metavar='ref', type=str,help='Path to the style reference text dataset.')
parser.add_argument('--model_input_name', default='../models/word-level/model_s2s.h5')
parser.add_argument('--encoder_model_input_name', default='../models/word-level/encoder_model_s2s.h5')
parser.add_argument('--decoder_model_input_name', default='../models/word-level/decoder_model_s2s.h5')
parser.add_argument('--sample_input_location', default="chaucer_texts/output/englishToChaucerModelTestingModernEnglishOnly.txt")
parser.add_argument('--batch_size', default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--latent_dim', default=256)
parser.add_argument('--num_samples', default= 10000)
args = parser.parse_args()

# Names of model files to load from
model_input_name = args.model_input_name
encoder_model_input_name = args.encoder_model_input_name
decoder_model_input_name = args.decoder_model_input_name
# Path to the sample texts to rewrite
sample_input_location = args.sample_input_location
# Path to the data txt file on disk.
data_path = args.data_path

# loads in model
model = load_model(model_input_name)

batch_size = args.batch_size  # Batch size for training.
epochs = args.epochs  # Number of epochs to train for.
latent_dim = args.latent_dim  # Latent dimensionality of the encoding space.
num_samples = args.num_samples  # Number of samples to train on.

# vectorises the data
# in this case, data is a mapping between the bi-grams of words in
# Chaucerian English and the bi-grams of words in the associated modern translation
# for example, a doghter which that called was Sophie.	a daughter who was called Sophie.
# would output a mapping between (a doghter) -> (a daughter), (doghter which) -> (daughter who)
# etc

# list of all lines of text for the input
# for example, "a doghter which that called was Sophie."
input_texts = []
# list of all lines of text for the target
# for example, "a daughter who was called Sophie."
target_texts = []
# list of all lines of texts from input as a collection of words
input_texts_as_words = []
# list of all lines of texts from target as a collection of words
target_texts_as_words = []
# list of all complete words in the text for the input
input_words = set()
# list of all complete words in the text for the target
target_words = set()
# set of all ngrams of texts for the input sentences
# (a doghter), (a yong)
input_ngrams = []
# set of all ngrams of texts for the target sentences
# (a daughter), (a young)
target_ngrams = []

# open the text file and gets all words/ngrams
with open(data_path, 'r', encoding="utf-8") as f:
    lines = f.read().split('\n')
# for each line in the text file
for line in lines[: min(num_samples, len(lines) - 1)]:
    # split the line by the tab word
    input_text, target_text = line.split('\t')
    #target_text = '\t' + target_text + '\n'
    # add the first half to input texts, then the second half to the target texts
    input_texts.append(input_text)
    target_texts.append(target_text)
    # breaks a string into a collection of words
    input_line_words = textinformation.clean_sentence_as_words(input_text)
    target_line_words = textinformation.clean_sentence_as_words(target_text)
    # removes any stop words from collection
    input_line_words = textinformation.remove_stop_words(input_line_words)
    target_line_words = textinformation.remove_stop_words(target_line_words)
    # adds sentences as words to collections
    input_texts_as_words.append(input_line_words)
    target_texts_as_words.append(target_line_words)
    # adds all unique words to the collection of words for input and target
    for word in input_line_words:
        if word not in input_words:
            input_words.add(word)
    for word in target_line_words:
        if word not in target_words:
            target_words.add(word)

# generates the information about the maximum size of sequences that can be generated
input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
# size of the vocabulary of words for the encoder
num_encoder_tokens = len(input_words)
# size of the vocabulary of words for the decoder
num_decoder_tokens = len(target_words)
# maximum possible vector size for the encoder
# in this case, determined as the longest sentence in number of words
max_encoder_seq_length = max([len(txt) for txt in input_texts_as_words])
# maximum possible vector size for the decoder
max_decoder_seq_length = max([len(txt) for txt in target_texts_as_words])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# creates a dictionary mapping a word to its positions
input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])

# creates numpy arrays containing information about where words appear in the sentences
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
# creates empty array which contains the information passed to
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
# creates empty array which will store the positions of words for the decoder output
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# populates the numpy arrays based on the position of words
for i, (input_text, target_text) in enumerate(zip(input_texts_as_words, target_texts_as_words)):
# updates encoder_input array with this word's position in input and target
    for t, word in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[word]] = 1.
    # updates decoder_input array with this word's position in
    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[word]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start word.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# using inference mode to retrieve predicted decoder states
encoder_model = load_model(encoder_model_input_name)
decoder_model = load_model(decoder_model_input_name)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_word_index = dict((i, word) for word, i in target_token_index.items())

# takes a string as an input and returns a numpy array
def encode_string_to_sequence(input_string):
    # create empty numpy array the same length as the input sentence
    converted_string = np.zeros(
        (len(input_string), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    # for every word in the string:
    for count, word in enumerate(input_string):
        # find the associated numpy array from the encoder_input_data
        # otherwise, select a random value from the numpy array
        # avoids errors if the sample sentence uses a word not seen before
        # TODO: replace this with an approach that preserves more of the sentence meaning
        try:
            converted_string[count] = encoder_input_data[input_token_index[word]]
        except KeyError:
            random_key_word = (random.choice(list(input_token_index)))
            converted_string[count] = encoder_input_data[input_token_index[random_key_word]]
    # return the combined numpy arrays
    return converted_string

# turns a numpy array into a sequence of words
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    # decoded sentence string
    decoded_sentence = ''
    while not stop_condition:
        # creates a list of output_tokens based on the prediction of the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        # gets the index as the indice of the largest element
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # gets the word from the word index dictionary
        sampled_word = reverse_target_word_index[sampled_token_index]
        # adds that word to the decoded sentence
        decoded_sentence += sampled_word + " "

        # Exit condition: either hit max length or find stop word.
        if (sampled_word == '\n' or
           len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
    return decoded_sentence

# passing in a new sentence from a text file unrelated to the original set
print("Attempting " + sample_input_location + " as input...")
with open(sample_input_location, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for item in lines:
        print ("Input sentence: ", item)
        item = textinformation.clean_sentence_as_words(item)
        item = textinformation.remove_stop_words(item)
        input_seq = encode_string_to_sequence(item)
        decoded_sentence = decode_sequence(input_seq)
        print("Decoded sentence: ", decoded_sentence)
        print("_____________________________________")
