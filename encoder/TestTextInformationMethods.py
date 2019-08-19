'''
Unit tests for TextInformation methods.
'''
import unittest
import TextInformation as ti

class TestTextInformationMethods(unittest.TestCase):
    # testing if cleaning a sentence works as expected
    def test_clean_sentence_as_words(self):
        sentence = "This: is; a. test- sentence!"
        sentence = ti.clean_sentence_as_words(sentence)
        self.assertEqual(sentence, ["This","is","a","test","sentence"])

    # testing if stop words are removed from a set correctly
    def test_remove_stop_words(self):
        sentence = ["This","is","a","test","sentence"]
        sentence = ti.remove_stop_words(sentence)
        self.assertEqual(sentence, ["This", "test", "sentence"])

    # testing if n grams are generated correctly
    def test_ngrams(self):
        sentence = ["This", "test", "sentence"]
        onegrams = ti.ngrams(sentence, 1)
        self.assertEqual(onegrams, [["This"], ["test"], ["sentence"]])
        bigrams = ti.ngrams(sentence, 2)
        self.assertEqual(bigrams, [["This", "test"], ["test", "sentence"]])
        trigrams = ti.ngrams(sentence, 3)
        self.assertEqual(trigrams, [["This", "test", "sentence"]])
        # checking if n-gram windows bigger than the sentence are rounded down
        quadrams = ti.ngrams(sentence, 4)
        self.assertEqual(quadrams, [["This", "test", "sentence"]])

        # testing if the prepare string method works correctly
    def test_prepare_string(self):
        sentence = "This: is; a. test- sentence!"
        prepared_string = ti.prepare_string(sentence)
        self.assertEqual(prepared_string, ["This", "test", "sentence"])
        no_punc_sentence = "This is a test sentence"
        prepared_no_punc = ti.prepare_string(no_punc_sentence)
        self.assertEqual(prepared_no_punc, ["This", "test", "sentence"])

    def test_ngram_to_sentence(self):
        sentence = "This is a test sentence"
        prepared_string = ti.prepare_string(sentence)
        self.assertEqual(prepared_string, ["This", "test", "sentence"])
        ngrams = ti.ngrams(prepared_string, 3)
        rebuilt_sentence = ti.ngram_to_sentence(ngrams)
        self.assertEqual("This test sentence", rebuilt_sentence)
        chaucer_sentence = "Then I conclude thus, since God of heaven"
        prepared_chaucer = ti.prepare_string(chaucer_sentence)
        self.assertEqual(prepared_chaucer, ["Then", "I", "conclude", "God", "heaven"])
        chaucer_ngrams = ti.ngrams(prepared_chaucer, 3)
        rebuilt_chaucer = ti.ngram_to_sentence(chaucer_ngrams)
        self.assertEqual("Then I conclude God heaven", rebuilt_chaucer)

if __name__ == '__main__':
    unittest.main()
