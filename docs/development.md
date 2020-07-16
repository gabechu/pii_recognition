# Setting Up a Development Environment


## Table of contents

1. [Installing the Zendesk App Tools (ZAT)](#dev-zat)
2. [Setting up the Python environment](#dev-python)
3. [Lauching the Flask app](#dev-flask)
4. [Getting browser connected to the Zendesk instance](#dev-instance)


## Installing the Zendesk App Tools (ZAT) <a name='dev-zat'></a>
ZAT is used for developing Zendesk apps and it bridges communications between your local (or remote) app and your Zendesk product instance.

To get it running, first you must install Ruby since ZAT is a Ruby gem.

Then, install `rake` a build automation tool.
```sh
gem install rake
```

Next, install ZAT
```sh
gem install zendesk_apps_tools
```

Navigate to **app_local** directory it contains all necessary files for any ZAT commands to run. Here, start a local HTTP server and then you can talk to your Zendesk product instance.
``` sh
zat server
```

If you want to customise the ZAT service, use **manifest.json** file. This file specifies one or more locations in one or more Zendesk products. In this project, we will build a Support app served at `localhost:8080/sidebar`.
```
...

  "location": {
    "support": {
      "ticket_sidebar": "http://127.0.0.1:8080/sidebar"
    }
  },

...

```

## Setting up the Python environment <a name='dev-python'></a>
Install Python 3.7 and pip install `poetry` package. `poetry` is a dependency management tool.
```
pip install --user poetry
```
Install ML libraries dependencies and Flask using `poetry`. Make sure the installation happens in the root directory.
```sh
poetry install
```
Activate the virtual environment.
```
poetry shell
```

## Launching the Flask app <a name='dev-flask'></a>
Run the application on localhost at port 8080 under directory `app_remote`.
```sh
flask run -p 8080
```

## Getting browser connecting to the Zendesk instance <a name='dev-instance'></a>
Open your favorite browser and type `https://z3n-numbat-piiredaction.zendesk.com/` in the location bar and hit Enter! You are now in the Support instance. Append `?zat=true` to the end of the page url and open a ticket, you shall be seeing the app running in the sidebar. You may have to ask @gabechu for access to the instance.

## Troubleshooting
- Is your ZAT server running?
- Is your Flask app running at port 8080?
- Have you connected to the correct instance at `https://z3n-numbat-piiredaction.zendesk.com/`?
- Have you appended to append `?zat=true` to the page url?
- Getting troubles installing ZAT, see [this page](https://develop.zendesk.com/hc/en-us/articles/360001075048-Installing-and-using-the-Zendesk-apps-tools) for details.