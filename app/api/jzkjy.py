from flask import Blueprint
from flask import jsonify
# from ..jz.test import test_1


jzkjy_bp = Blueprint('jzkjy', __name__)


@jzkjy_bp.route('/station/run')
def run():
    pass
    # df = test_1()
    # return jsonify(df)

