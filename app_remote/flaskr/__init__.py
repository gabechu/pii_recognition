from typing import Optional

from flask import Flask, render_template, Response


def create_app(test_config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    @app.route("/sidebar")
    def sidebar() -> str:
        return render_template("sidebar.html")

    @app.route("/replace-ticket-id", methods=["PUT"])
    def update_ticket_id() -> Response:
        ...
        # stub will be implemented

    return app
