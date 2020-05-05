from .data_path import DataPath
from pytest import raises


def test_DataPath():
    actual = DataPath("pii_recognition/datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"
    assert actual.reader_name == "ConllReader"

    actual = DataPath("/datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"
    assert actual.reader_name == "ConllReader"

    actual = DataPath("datasets/conll2003/eng.testa")
    assert actual.valid is True
    assert actual.data_name == "conll"
    assert actual.version == "2003"
    assert actual.reader_name == "ConllReader"

    actual = DataPath("datasets/wnut2017/emerging.test.annotated")
    assert actual.valid is True
    assert actual.data_name == "wnut"
    assert actual.version == "2017"
    assert actual.reader_name == "WnutReader"

    actual = DataPath("datasets/fake2020/en.test")
    assert actual.valid is True
    assert actual.data_name == "fake"
    assert actual.version == "2020"
    with raises(NameError) as err:
        getattr(actual, "reader_name")
    assert str(err.value) == "No reader found to process fake dataset"

    actual = DataPath("broken_path")
    assert actual.valid is False
    with raises(AttributeError):
        getattr(actual, "data_name")
    with raises(AttributeError):
        getattr(actual, "version")
