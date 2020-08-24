# PII Redaction App

PII redaction app runs on Zendesk Support.

This app aims to help agents and admins automate the process of ticket de-identification.

This folder contains all the necessary files for this app to run on a Zendesk Support product. You may need [ZAT](https://developer.zendesk.com/apps/docs/developer-guide/zat) for local development. ZAT will allow you to validate and testing this app by simply typing `zat validate` and `zat server`. In the end, use command `zat package` to wrap up the folder to a zip file and upload to the Support instance and install it as a private app.

### The following information is displayed:
Content will be provide soon.

* Dropdown box to select a machine learning model
* Detect button to identify PII data in comments
* Callout box to render PII findings
* Reset button to clear out contents in callout box
* Redact button to redact contents in callout box

Please submit bug reports to [Insert Link](https://github.com/gabechu/pii_recognition/issues). Pull requests are welcome.

### Screenshot(s):
![App Demo](../assets/app_demo.gif?raw=true)
