from flask import Flask
from routes import predictor, root_routes, file_manager
from gevent.wsgi import WSGIServer
from models.predicted_prices import db
import logging

app = Flask(__name__, static_url_path='/public')


def init_app():
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
    # app.run(host='0.0.0.0', port=7000) # debug server
    print('Starting server')
    http_server = WSGIServer(('', 7000), app)
    http_server.serve_forever()
