let awsClient;

if (typeof AWS === "undefined") {
  throw new Error("Could not initiate AWS SDK");
} else {
  awsClient = AWS;
  awsClient.config.region = "us-west-2"; // Region
  awsClient.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: "placeholder",
  });
}

const comprehendClient = new awsClient.Comprehend();

export { awsClient, comprehendClient };
