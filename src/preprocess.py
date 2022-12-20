import pandas as pd
import re 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


class Preprocessing:
    def __init__(self, args):
        self.filepath = "data/train.csv"
        self.maxwords = args.max_words #choose 100 most frequent word in vocab
        self.maxlen = args.max_len #maximum length of a sequence
        self.test_size = args.test_size

    def load_data(self):
        #drop unnecessary column
        df = pd.read_csv(self.filepath)
        df.drop(["id","keyword","location"], inplace=True, axis=1)

        #define data to train and label
        train = df['text'].values
        label = df['target'].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(train, label, test_size=self.test_size)

        for i in self.X_train:
            i = re.sub(r"[^A-Za-z0-9]+","",i)
    
    def tokenization(self):
        self.tokens = Tokenizer(num_words=self.maxwords)
        self.tokens.fit_on_texts(self.X_train)

    def sequence_to_token(self,input):
        sequences = self.tokens.texts_to_sequences(input)
        return pad_sequences(sequences, maxlen = self.maxlen)


        