from flask import Flask, render_template, url_for
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('homepage.html')


if __name__ == "__main__":
    app.run(debug=True)
