# script to turn a plain text Gutenberg book into a csv of lines and authors
import sys
# argument parsing library
import argparse
# NLP annotation library
import spacy
# Python mongoDB connector
import pymongo
# loads English dictionary into spaCy
nlp = spacy.load("en")

def get_my_string(inputFn, author):
    try:
        with open(inputFn) as inputFileHandle:
            flist = inputFileHandle.readlines()
            flist = [line for line in flist if line.strip().split()]
            parsing = False
            id = 0
            trainingDataCount = 0
            for line in flist:
                if "*** END OF THIS PROJECT GUTENBERG EBOOK" in line  or "*** END OF THE PROJECT GUTENBERG EBOOK" in line:
                    parsing = False
                    break
                if parsing:
                    # NLP processing
                    doc = nlp(line)
                    tokens = []
                    pos = []
                    tags = []
                    dependencies = []
                    for token in doc:
                        tokens.append(str(token))
                        pos.append(token.pos_)
                        tags.append(token.tag_)
                        dependencies.append(token.dep_)
                    # outputs contents to database
                    contents =  {"author": author, "line" : line, "tokens": tokens, "pos" : pos, "tags" : tags, "dependencies" : dependencies}
                    print (contents)
                    if (trainingDataCount % 10 == 0):
                        x = testing.insert_one(contents)
                    else:
                        x = training.insert_one(contents)
                    trainingDataCount += 1

                if "*** START" in line:
                    parsing = True

    except IOError:
        sys.stderr.write( "[myScript] - Error: Could not open %s\n" % (inputFn) )
        sys.exit(-1)

parser = argparse.ArgumentParser(description='Processes and adds the contents of a Project Gutenbetg text file to a MongoDB.')
parser.add_argument('input', help='file to process')
parser.add_argument('author', help='author of contents of input')

args = parser.parse_args()

myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["text-dataset"]
training = mydb["sample-training"]
testing = mydb["sample-testing"]

get_my_string(args.input, args.author)
