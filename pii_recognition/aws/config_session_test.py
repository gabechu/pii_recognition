from moto import mock_cognitoidentity

from pii_recognition.aws.config_session import config_cognito_session


@mock_cognitoidentity
def test_create_cognito_session():
    session = config_cognito_session(
        identity_pool_id="us-west-2:11aa1111-1aa1-1a1a-11aa-11aa1aaaa111",
        region="us-west-2")

    credentials = session.get_credentials()
    assert session.region_name == "us-west-2"
    assert credentials.access_key == "TESTACCESSKEY12345"
    assert credentials.secret_key == "ABCSECRETKEY"
