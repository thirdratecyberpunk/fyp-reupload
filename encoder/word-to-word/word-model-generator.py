#!/usr/bin/python
'''Generates the model for the n-gram based model.'''

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import argparse
import spacy
from datetime import datetime
import string
import sys
import TextInformation as textinformation

currentdatetime = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

# loads in the English dictionary into spacy
nlp = spacy.load('en')

parser = argparse.ArgumentParser(description='Generates models trained on words and ngrams of an interlinear translation.')
parser.add_argument('data_path', metavar='ref', type=str,help='Path to the style reference text dataset.')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--latent_dim', default=256)
parser.add_argument('--num_samples', default= 10000)
parser.add_argument('--model_output_name', default='../models/word-level-models/model_s2s' + str(currentdatetime) + '.h5')
parser.add_argument('--encoder_model_output_name', default='../models/word-level-models/encoder_model_s2s' + str(currentdatetime) + '.h5')
parser.add_argument('--decoder_model_output_name', default='../models/word-level-models/decoder_model_s2s' + str(currentdatetime) + '.h5')

args = parser.parse_args()

data_path = args.data_path
num_samples = args.num_samples
batch_size = args.batch_size  # Batch size for training.
epochs = args.epochs  # Number of epochs to train for.
latent_dim = args.latent_dim  # Latent dimensionality of the encoding space.
num_samples = args.num_samples # Number of samples to train on.
# Path to the data txt file on disk.
data_path = args.data_path
# Path to the save location of the models.
model_output_name = args.model_output_name
encoder_model_output_name = args.encoder_model_output_name
decoder_model_output_name = args.decoder_model_output_name

# TODO: move this process to a separate class
# vectorises the data
# in this case, data is a mapping between the bi-grams of words in
# Chaucerian English and the bi-grams of words in the associated modern translation
# for example, a doghter which that called was Sophie.	a daughter who was called Sophie.
# would output a mapping between (a doghter) -> (a daughter), (doghter which) -> (daughter who)
# etc

# TODO: include bigram data, not just word level

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
    # split the line by the tab character
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
input_token_index = dict([(char, i) for i, char in enumerate(input_words)])
target_token_index = dict([(char, i) for i, char in enumerate(target_words)])

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
# updates encoder_input array with this character's position in input and target
    for t, word in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[word]] = 1.
    # updates decoder_input array with this character's position in
    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[word]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
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

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save(model_output_name)

# defining the encoder model states
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save(encoder_model_output_name)

# defining the decoder model states
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

decoder_model.save(decoder_model_output_name)
