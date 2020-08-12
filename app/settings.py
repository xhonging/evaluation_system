import os
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BasicConfig:
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(BasicConfig):
    DEBUG = True
    # SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost:3306/aps_py'


class TestingConfig(BasicConfig):
    TESTING = True
    DEBUG = True
    # SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost:3306/test_aps'


class ProductionConfig(BasicConfig):
    DEBUG = False
    # SQLALCHEMY_DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost:3306/aps_py'


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# SWAGGER_TITLE = 'API'
# SWAGGER_DESC = '接口文档'
# SWAGGER_HOST = 'http://127.0.0.1:5000/api/v1'
ERROR_LOG = "../logs/error.logs"
INFO_LOG = "../logs/info.logs"

