from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
from nltk import PorterStemmer, word_tokenize
ps = PorterStemmer()


class Vocab(object):
    def __init__(self, concepts, is_stem):
        super(Vocab, self).__init__()
        """ self-organized vocabulary
        """
        self.is_stem = is_stem

        # dual task concept (dual task)
        self.concepts, concept_size = concepts, len(concepts)
        self.concept_idx = {self.stem(tok): i+1 for i, tok in enumerate(self.concepts)}  # bias 1 for [pad]
        self.concept_size_en = max([len(self.concepts), len(self.concept_idx)]) + 1  # bias 1 for [pad]
        self.pad, self.pad_idx = "[pad]", 0
        print("Vocab: STEM:", self.is_stem)
        print("Vocab: Concept size:", len(self.concepts))
        print("Vocab: Concept idx size:", len(self.concept_idx))

        # bag of words (sklearn)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.concepts)
        self.names = self.vectorizer.get_feature_names()
        self.concept_size_de = len(self.names)
        print("Vocab: Vectorizer name size:", len(self.names))

    def text_to_bow(self, text):
        if isinstance(text, str):
            res = self.vectorizer.transform([text]).toarray()
        elif isinstance(text, list):
            res = self.vectorizer.transform(text).toarray()
        else:
            raise NotImplementedError
        return res

    def text_to_tok_id(self, text):
        text = re.sub(r"[^A-Za-z0-9]", " ", text.lower())
        tokens = word_tokenize(text)
        tokens = [self.stem(tok) for tok in tokens]
        tok_ids = [self.concept_idx[tok] if tok in self.concept_idx else self.pad_idx for tok in tokens]
        return torch.tensor(tok_ids)

    def stem(self, x):
        if self.is_stem:
            return ps.stem(x)
        return x

