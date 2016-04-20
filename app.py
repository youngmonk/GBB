from flask import Flask, send_from_directory
from routes import predictor
import logging

app = Flask(__name__, static_url_path='/public')
UPLOAD_FOLDER = ''

app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
    return send_from_directory('public', 'index.html')


@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('public', path)

# initialize all routes
predictor.init(app=app)

if __name__ == "__main__":
    app.run()
