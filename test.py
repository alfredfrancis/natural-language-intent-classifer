import unittest
from sentenceClassifer import train,predict
import os

class TestClassifer(unittest.TestCase):
    def setUp(self):
        self.X = ["hello friend",
                 "haai",
                 "hey",
                 " ",
                 "hii",
                 "hey",
                 ""]

        self.y = ["greeting",
             "greeting",
             "greeting",
             "fallback",
             "greeting",
             "greeting",
             "fallback"]

        self.PATH = "test.model"

        train(self.X, self.y, outpath=self.PATH, verbose=False)

    def test_classifer(self):
        for sentence, label in zip(self.X,self.y):
            self.assertEqual(predict(sentence,self.PATH)["class"],label)
        os.remove(self.PATH)

if __name__ == "__main__": unittest.main()