from flask import Flask, render_template

def create_app():
    app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

    # Homepage
    @app.route("/")
    def home():
        return render_template("index.html")

    return app
