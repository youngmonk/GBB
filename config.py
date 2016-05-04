class Config(object):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:zoomo123@localhost:5432/gozoomo'
    SQLALCHEMY_POOL_SIZE = 10
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    POOL_SIZE = 10
    UPLOAD_FOLDER = ''


class ProductionConfig(Config):
    DEBUG = False
