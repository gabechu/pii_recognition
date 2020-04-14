from .data_path import DataPath
from pytest import raises


def test_DataPath():
    actual = DataPath("pii_recognition/datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"

    actual = DataPath("/datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"

    actual = DataPath("datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"

    actual = DataPath("broken_path")
    assert actual.valid is False
    with raises(AttributeError):
        getattr(actual, "data_name")
    with raises(AttributeError):
        getattr(actual, "version")
