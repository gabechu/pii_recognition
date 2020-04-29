from .token_schema import Token
from .tokenisers import TreebankWordTokeniser


def test_nltk_word_tokenizer():
    treebank_word_tokeniser = TreebankWordTokeniser()

    text = "This is a test."
    actual = treebank_word_tokeniser.tokenise(text)
    assert actual == [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("a", 8, 9),
        Token("test", 10, 14),
        Token(".", 14, 15),
    ]

    text = "I'm here"
    actual = treebank_word_tokeniser.tokenise(text)
    assert actual == [Token("I", 0, 1), Token("'m", 1, 3), Token("here", 4, 8)]
