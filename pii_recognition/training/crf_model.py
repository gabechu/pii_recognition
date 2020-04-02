# Copyright
# https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
from itertools import chain

import pycrfsuite
from nltk.corpus.reader import ConllCorpusReader
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

train = ConllCorpusReader(
    "datasets/conll2003", "eng.train", ["words", "pos", "ignore", "chunk"]
)
test = ConllCorpusReader(
    "datasets/conll2003", "eng.testb", ["words", "pos", "ignore", "chunk"]
)

train_sents = list(train.iob_sents())
test_sents = list(test.iob_sents())


def word2features(sent, i):
    # remove postag
    word = sent[i][0]
    # postag = sent[i][1]
    features = [
        "bias",
        "word.lower=" + word.lower(),
        "word[-3:]=" + word[-3:],
        "word[-2:]=" + word[-2:],
        "word.isupper=%s" % word.isupper(),
        "word.istitle=%s" % word.istitle(),
        "word.isdigit=%s" % word.isdigit(),
        # 'postag=' + postag,
        # 'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        # postag1 = sent[i-1][1]
        features.extend(
            [
                "-1:word.lower=" + word1.lower(),
                "-1:word.istitle=%s" % word1.istitle(),
                "-1:word.isupper=%s" % word1.isupper(),
                # '-1:postag=' + postag1,
                # '-1:postag[:2]=' + postag1[:2],
            ]
        )
    else:
        features.append("BOS")

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        # postag1 = sent[i+1][1]
        features.extend(
            [
                "+1:word.lower=" + word1.lower(),
                "+1:word.istitle=%s" % word1.istitle(),
                "+1:word.isupper=%s" % word1.isupper(),
                # '+1:postag=' + postag1,
                # '+1:postag[:2]=' + postag1[:2],
            ]
        )
    else:
        features.append("EOS")
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params(
    {
        "c1": 1.0,  # coefficient for L1 penalty
        "c2": 1e-3,  # coefficient for L2 penalty
        "max_iterations": 50,  # stop earlier
        # include transitions that are possible, but not observed
        "feature.possible_transitions": True,
    }
)

trainer.train("conll2003-en.crfsuite")

tagger = pycrfsuite.Tagger()
tagger.open("conll2003-en.crfsuite")

example_sent = test_sents[0]
print(" ".join(sent2tokens(example_sent)), end="\n\n")
print("Predicted:", " ".join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", " ".join(sent2labels(example_sent)))


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"O"}
    taglist = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in taglist],
        target_names=taglist,
    )


y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))
