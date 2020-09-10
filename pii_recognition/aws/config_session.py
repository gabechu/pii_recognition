import boto3
from boto3.session import Session


def config_cognito_session(identity_pool_id: str, region: str) -> Session:
    client = boto3.client(service_name="cognito-identity", region_name=region)

    response = client.get_id(IdentityPoolId=identity_pool_id)
    identity_id = response["IdentityId"]

    response = client.get_credentials_for_identity(IdentityId=identity_id)
    secretKey = response['Credentials']['SecretKey']
    accessKey = response['Credentials']['AccessKeyId']
    sessionToken = response['Credentials']['SessionToken']

    return Session(aws_access_key_id=accessKey,
                   aws_secret_access_key=secretKey,
                   aws_session_token=sessionToken,
                   region_name=region)
