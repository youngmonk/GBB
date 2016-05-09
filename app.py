from flask import Flask
from routes import predictor, root_routes, file_manager
from gevent.wsgi import WSGIServer
from models.predicted_prices import db
import os
import logging

app = Flask(__name__, static_url_path='/public')
env = os.environ.get('env', 'DEV') # default env is DEV. For prod set an environment variable env as PROD

def init_app():
    if env == 'PROD':
        print('Configuring for prod')
        app.config.from_object('config.ProductionConfig')
    else:
        print('Configuring for develop')
        app.config.from_object('config.Config')
    db.init_app(app=app)


# initialize all routes
def init_routes():
    root_routes.init(app=app)
    predictor.init(app=app)
    file_manager.init(app=app)


if __name__ == "__main__":
    init_app()
    init_routes()
    print('Starting server')
    if env == 'PROD':
        http_server = WSGIServer(('', 7000), app)
        http_server.serve_forever()
    else:
        app.run(host='0.0.0.0', port=7000) # debug server

