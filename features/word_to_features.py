def word2features(tokenised_sentence, i):
    word = tokenised_sentence[i]

    features = [
        "bias",
        "word.lower=" + word.lower(),
        "word[-3:]=" + word[-3:],
        "word[-2:]=" + word[-2:],
        "word.isupper=%s" % word.isupper(),
        "word.istitle=%s" % word.istitle(),
        "word.isdigit=%s" % word.isdigit(),
    ]
    if i > 0:
        word1 = tokenised_sentence[i - 1]
        features.extend(
            [
                "-1:word.lower=" + word1.lower(),
                "-1:word.istitle=%s" % word1.istitle(),
                "-1:word.isupper=%s" % word1.isupper(),
            ]
        )
    else:
        features.append("BOS")

    if i < len(tokenised_sentence) - 1:
        word1 = tokenised_sentence[i + 1]

        features.extend(
            [
                "+1:word.lower=" + word1.lower(),
                "+1:word.istitle=%s" % word1.istitle(),
                "+1:word.isupper=%s" % word1.isupper(),
            ]
        )
    else:
        features.append("EOS")
    return features
