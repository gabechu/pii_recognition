from pii_recognition.labels.schema import Entity
from pytest import fixture


@fixture(scope="module")
def recogniser():
    from pii_recognition.recognisers import registry as recogniser_registry

    recogniser_name = "ComprehendRecogniser"
    recogniser_params = {
        "supported_entities": [
            "BANK_ACCOUNT_NUMBER",
            "BANK_ROUTING",
            "CREDIT_DEBIT_NUMBER",
            "CREDIT_DEBIT_CVV",
            "CREDIT_DEBIT_EXPIRY",
            "PIN",
            "NAME",
            "ADDRESS",
            "PHONE",
            "EMAIL",
            "AGE",
            "USERNAME",
            "PASSWORD",
            "URL",
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
            "IP_ADDRESS",
            "MAC_ADDRESS",
            "SSN",
            "PASSPORT_NUMBER",
            "DRIVER_ID",
            "DATE_TIME",
        ],
        "supported_languages": ["en"],
        "model_name": "pii",
    }

    return recogniser_registry.create_instance(recogniser_name, recogniser_params)


def test_BANK_ACCOUNT_NUMBER(recogniser):
    ...


def test_BANK_ROUTING(recogniser):
    text = (
        "When initiating a domestic wire transfer to Huntington, use the "
        "routing number 044000024."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 044000024
    assert Entity("BANK_ROUTING", 79, 88) in actual


def test_CREDIT_DEBIT_NUMBER(recogniser):
    text = (
        "Please update billing addrress with Markt 84, MÜLLNERN 9123 for "
        "this card: 5550253262199449."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 5550253262199449
    assert Entity("CREDIT_DEBIT_NUMBER", 75, 91) in actual


def test_CREDIT_DEBIT_CVV(recogniser):
    ...


def test_CREDIT_DEBIT_EXPIRY(recogniser):
    text = "What should I do when my card expires? My card expires 06/21."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 0621
    assert Entity("CREDIT_DEBIT_EXPIRY", 55, 60) in actual


def test_PIN(recogniser):
    ...


def test_NAME(recogniser):
    text = "A tribute to Joshua Lewis – sadly, she wasn't impressed."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: Joshua Lewis
    assert Entity("NAME", 13, 25) in actual


def test_ADDRESS(recogniser):
    text = "The address of Balefire Global is Valadouro 3, Ubide 48145."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: Valadouro 3, Ubide 48145
    assert Entity("ADDRESS", 34, 58) in actual


def test_PHONE(recogniser):
    text = (
        "Please have the manager call me at 0378 8718408 I'd like to join accounts "
        "with ms. Caroline."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 0378 8718408
    assert Entity("PHONE", 35, 47) in actual


def test_EMAIL(recogniser):
    text = (
        "Please transfer all funds from my account to this "
        "hackers' VidoslavBabic@rhyta.com."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: VidoslavBabic@rhyta.com
    assert Entity("EMAIL", 59, 82) in actual


def test_AGE(recogniser):
    ...


def test_USERNAME(recogniser):
    text = "I have a seller account and my username is water_ionizers_and_f."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: water_ionizers_and_f
    assert Entity("USERNAME", 43, 63) in actual


def test_PASSWORD(recogniser):
    text = (
        "An example of a strong password is Cartoon-Duck-14-Coffee-Glvs. "
        "It is long, contains uppercase letters, lowercase letters, numbers, "
        "and special characters."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: Cartoon-Duck-14-Coffee-Glvs
    assert Entity("PASSWORD", 35, 62) in actual


def test_URL(recogniser):
    text = "Just posted a photo http://premiumbot.com.cy/gentlemudNpca4."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: http://premiumbot.com.cy/gentlemudNpca4
    assert Entity("URL", 20, 59) in actual


def test_AWS_ACCESS_KEY(recogniser):
    ...


def test_AWS_SECRET_KEY(recogniser):
    ...


def test_IP_ADDRESS(recogniser):
    text = (
        "I can't browse to your site, keep getting address "
        "c4c4:9bac:38a3:886:f173:826c:d16d:e730 blocked error."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: c4c4:9bac:38a3:886:f173:826c:d16d:e730
    assert Entity("IP_ADDRESS", 50, 88) in actual


def test_MAC_ADDRESS(recogniser):
    text = (
        "A MAC address consists of six sets of two characters, each separated by a "
        "colon. 00:1B:44:11:3A:B7 is an example of a MAC address."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 00:1B:44:11:3A:B7
    assert Entity("MAC_ADDRESS", 81, 98) in actual


def test_SSN(recogniser):
    text = "His social security number is 831-61-5012."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 831-61-5012
    assert Entity("SSN", 30, 41) in actual


def test_PASSPORT_NUMBER(recogniser):
    text = (
        "Can you help me check my flight reservation. I've booked with "
        "passport M0993353."
    )
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: M0993353
    assert Entity("PASSPORT_NUMBER", 71, 79) in actual


def test_DRIVER_ID(recogniser):
    ...


def test_DATE_TIME(recogniser):
    text = "Please tell me your date of birth. It's 10/19/1975."
    actual = recogniser.analyse(text, recogniser.supported_entities)

    # Entity: 10/19/1975
    assert Entity("DATE_TIME", 40, 50) in actual
