'''
Sequence to sequence example in Keras (character-level).
'''
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import argparse
import sys

# defining command line arguments
parser = argparse.ArgumentParser(description='Character level sequence to sequence model.')
parser.add_argument('data_path', metavar='ref', type=str,help='Path to the style reference text dataset.')
parser.add_argument('--model_input_name', default='models/model_s2s.h5')
parser.add_argument('--encoder_model_input_name', default='models/encoder_model_s2s.h5')
parser.add_argument('--decoder_model_input_name', default='models/decoder_model_s2s.h5')
parser.add_argument('--sample_input_location', default="input/test.txt")
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

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# sorts the input characters and determines the size of the numpy arrays
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# creates an index mapping characters to their position
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# creates numpy arrays storing vectorised versions of the input and target data
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# populates the decoder input data by combining the inputs and targets
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

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
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# using inference mode to retrieve predicted decoder states
encoder_model = load_model(encoder_model_input_name)
decoder_model = load_model(decoder_model_input_name)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# takes a string as an input and returns a numpy array
def encode_string_to_sequence(input_string):
    # create empty numpy array the same length as the input sentence
    converted_string = np.zeros(
        (len(input_string), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    # for every character in the string:
    for count, char in enumerate(input_string):
        # find the associated numpy array from the encoder_input_data
        converted_string[count] = encoder_input_data[input_token_index[char]]
        # add this to the new numpy array
    # return the combined numpy arrays
    return converted_string

# turns a numpy array into a sequence of characters
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

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
        # gets the character from the character index dictionary
        sampled_char = reverse_target_char_index[sampled_token_index]
        # adds that character to the decoded sentence
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
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
        input_seq = encode_string_to_sequence(item.strip())
        decoded_sentence = decode_sequence(input_seq)
        print ("Input sentence: ", item)
        print("Decoded sentence: ", decoded_sentence)
