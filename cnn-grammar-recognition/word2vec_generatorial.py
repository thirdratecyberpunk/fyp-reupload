import argparse
import word2vec
import numpy
import json
from spacy.vocab import Vocab
from spacy.vectors import Vectors
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import pprint as pp

def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

# defining command line arguments
parser = argparse.ArgumentParser(description='Neural style transfer between texts with Keras.')
parser.add_argument('style_reference_text_path', metavar='ref', type=str,help='Path to the style reference text dataset.')
parser.add_argument('--seed', default=1000, help="Seed for numpy random.")
parser.add_argument('--max_words', default=10000, help="Number of top words to load into model.")

args = parser.parse_args()

# sets numpy seed for reproducibility
numpy.random.seed(args.seed)

max_words = args.max_words

# loads content from given text file
print("Loading text from file " + args.style_reference_text_path + "...")
try:
    with open(args.style_reference_text_path) as inputFileHandle:
        flist = inputFileHandle.readlines()
        # splitting lines and removing newline characters
        flist = [line.strip() for line in flist]
        # removing any empty lists
        flist = list(filter(None, flist))
        # ignoring any project Gutenberg metadata
        train_x = []
        parsing = False
        for line in flist:
            if "*** END OF THIS PROJECT GUTENBERG EBOOK" in line  or "*** END OF THE PROJECT GUTENBERG EBOOK" in line:
                parsing = False
                break
            # ignores any lines which only contain sonnet numbers
            if parsing and not line.isdigit():
                train_x.append(line)
            if "*** START" in line:
                parsing = True

except IOError:
    sys.stderr.write( "[myScript] - Error: Could not open %s\n" % (ref) )
    sys.exit(-1)

train_x = [str(x) for x in train_x]
# turning cleaned data into vectors
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(train_x)
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

allWordIndices = []
# for each sentence, change each token to the ID in the tokenizer's word index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)
# list of all sentences convered to index arrays
allWordIndices = numpy.asarray(allWordIndices)

# create one-hot matrices out of indexed sentences
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode="binary")
pp.pprint(train_x)

# TODO: create encoder for text
