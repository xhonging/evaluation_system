from flask import Flask
# from flask_bootstrap import Bootstrap
# from flask_migrate import Migrate
from app.settings import config, ERROR_LOG, INFO_LOG
from app.api.jjmj import jjmj_bp
from app.api.hgkj import hgkj_bp
from app.api.qdlsc import qdlsc_bp
from app.api.jzkjy import jzkjy_bp
# from flask_login import LoginManager
# from flasgger import Swagger, swag_from
# from config import SWAGGER_TITLE, SWAGGER_DESC, SWAGGER_HOST
# db = SQLAlchemy()
# bootstrap = Bootstrap()
# migrate = Migrate()
# 设置ip，端口号
# pycharm edit Additional Options: --host=0.0.0.0 --port=666 ，0.0.0.0表示对外开放，访问地址为电脑ip+url


def create_app(config_name):
    app = Flask(__name__)
    # swagger_config = Swagger.DEFAULT_CONFIG
    # swagger_config['title'] = SWAGGER_TITLE
    # swagger_config['description'] = SWAGGER_DESC
    # swagger_config['host'] = SWAGGER_HOST
    # Swagger(app, config=swagger_config)
    app.config.from_object(config[config_name])
    app.config["ERROR_LOG"] = ERROR_LOG
    app.config["INFO_LOG"] = INFO_LOG
    config[config_name].init_app(app)

    # db.init_app(app)
    # bootstrap.init_app(app)
    # migrate.init_app(app)

    register_blueprints(app)        # 注册蓝本
    register_shell_context(app)     # 注册shell上下文处理函数

    from .log import init_logging
    init_logging(app)

    return app


def register_blueprints(app):
    app.register_blueprint(jjmj_bp, url_prefix='/jjmj')
    app.register_blueprint(hgkj_bp, url_prefix='/hgkj')
    app.register_blueprint(qdlsc_bp, url_prefix='/qdlsc')
    app.register_blueprint(jzkjy_bp, url_prefix='/jzkjy')
    # from .. import blueprint as api_blueprint
    # app.register_blueprint(api_blueprint, url_prefix='/jjmj')


def register_shell_context(app):
    @app.shell_context_processor
    def make_shell_context():
        return dict(app=app)

