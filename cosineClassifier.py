import sys
import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CosineModel:
    def __init__(self, intentions=None, utterance_vector=None, vectorizer=None, logger=None):
        self.utterance_vector = utterance_vector
        self.vectorizer = vectorizer
        self.intentions = intentions
        self.logger = logger or logging.getLogger(__name__)
        self.training_df = None

    def train(self, training_df):
        try:
            vectorizer = TfidfVectorizer(stop_words=None)
            utterance_vector = vectorizer.fit_transform(training_df.loc[:, "Utterance"].values.astype("U"))
        except ValueError:
            self.logger.error('Invalid training data.', exc_info=True)
            sys.exit()

        self.training_df = training_df[["Intention"]]
        self.intentions = training_df.loc[:, "Intention"].values.astype("U")
        self.utterance_vector = utterance_vector
        self.vectorizer = vectorizer

    def save(self, model_location):
        if not os.path.exists(model_location):
            os.makedirs(model_location)

        with open(os.path.join(model_location, "tfidf_matrix.mtx"), "wb") as outfile:
            pickle.dump(self.utterance_vector, outfile, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(model_location, "vectorizer.pk"), "wb") as outfile:
            pickle.dump(self.vectorizer, outfile)
        self.training_df.to_csv(os.path.join(model_location, "intentions.csv"))

    @staticmethod
    def load(model_location):
        with open(os.path.join(model_location, 'tfidf_matrix.mtx'), 'rb') as infile:
            utterance_vector = pickle.load(infile)

        with open(os.path.join(model_location, 'vectorizer.pk'), 'rb') as f:
            vectorizer = pickle.load(f)
        intentions = pd.read_csv(os.path.join(model_location, 'intentions.csv'))
        return CosineModel(intentions=intentions['Intention'].values.astype("U"), utterance_vector=utterance_vector,
                           vectorizer=vectorizer)

    def predict(self, query,max_results = 10):
        query_vector = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(self.utterance_vector,
                                       query_vector)  # Measuring cosine similarity of query with training data.
        sim_with_each_intentions = list(zip(self.intentions, cosine_sim))
        sorted_similarity = sorted(sim_with_each_intentions, key=lambda x: x[1], reverse=True)  # Sorting based
        result = []
        count = 0
        if sorted_similarity[0][1][0] == 0.00:
            result.append({'intent': 'fallback', 'prob': 1.00})
        else:
            intention_set_debug = set()
            for (intention, sim) in sorted_similarity:
                if intention not in intention_set_debug and count < max_results:
                    result.append({'intent': intention, 'prob': (sim.tolist())[0]})
                    intention_set_debug.add(intention)
                    count += 1
        self.logger.info("Prediction completed")
        return result


if __name__ == "__main__":
    # training_data = [
    #     ["Greeting", "hello"],
    #     ["Greeting", "hey"],
    #     ["Greeting", "hi"],
    #     ["Goodbye", "bye"],
    #     ["Goodbye", "goodbye"]
    # ]
    #
    #
    # df = pd.DataFrame(training_data)

    df = pd.read_csv("utterances.csv")
    df.columns = ["Intention", "Utterance"]

    cosine = CosineModel()
    cosine.train(training_df=df)
    cosine.save("models")

    print(cosine.predict("Looking for chinese restaurents"))
