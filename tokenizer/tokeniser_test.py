from .tokeniser import nltk_word_tokenizer
from .token import Token


def test_nltk_word_tokenizer():
    text = "This is a test."
    actual = nltk_word_tokenizer(text)
    assert actual == [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("a", 8, 9),
        Token("test", 10, 14),
        Token(".", 14, 15),
    ]

    text = "I'm here"
    actual = nltk_word_tokenizer(text)
    assert actual == [Token("I", 0, 1), Token("'m", 1, 3), Token("here", 4, 8)]
