let client;

if (typeof ZAFClient === "undefined") {
  throw new Error("Could not initiate ZAFClient");
} else {
  client = ZAFClient.init();
}

export default client;
