'''
Class containing information about a model's vocabulary.
'''
import spacy
import string
nlp = spacy.load("en")

'''
Converts a string into a series of n-grams.
:param input: input string to convert
:param n: size of n-gram windows
:returns: a set containing n-grams of the sentence
'''
def ngrams(input, n):
  output = []
  # caps the ngram size as the length of the input
  if (n > len(input)):
      n = len(input)
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output

'''
Removes punctuation from a string and returns the string as a list of words.
:param input: input string to convert
:returns: list of words
'''

def clean_sentence_as_words(input):
    translator = str.maketrans('','', string.punctuation)
    output = input.strip().split()
    output = [word.translate(translator) for word in output]
    return output

'''
Removes any stop words from a list of words
:param input: input list to clean
:returns: list of words
'''
def remove_stop_words(input):
    return [word for word in input if word not in nlp.Defaults.stop_words]

'''
Processes a string, returning a cleaned sentence without any stop words
:param input: input string to clean
:returns: list of non-stop words
'''
def prepare_string(input):
    return remove_stop_words(clean_sentence_as_words(input))

'''
Processes a list of ngrams, returning a string representation of a sentence
For example, the trigram
[['Then', 'I', 'conclude'], ['I', 'conclude', 'God'], ['conclude', 'God', 'heaven']]
should be transformed into "Then I conclude God heaven"
:param input: list of ngrams to transform
:returns: string
'''
def ngram_to_sentence(input):
    ngram_words = []
    # add all words in the first ngram into the list
    for word in input[0]:
        ngram_words.append(word)
    previous_ngram = input[0]
    # for every remaining ngram
    for ngram in input[1:]:
        for word in ngram:
                # if the word is not in the previous n-gram, add it to the list
            if word not in previous_ngram:
                ngram_words.append(word)
        previous_ngram = ngram
    # convert the list of words into a sentence
    string_representation = ""
    for word in ngram_words[:-1]:
        string_representation += word + " "
    string_representation += ngram_words[-1]
    # return the sentence
    return string_representation
