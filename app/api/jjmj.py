from flask import Blueprint
from flask import jsonify
from ..jjmj.report import generate_report
from ..jjmj.reduce_power import get_reduce_data, get_reduce_picture
from ..jjmj.test import test_station
from ..jjmj.alarm_system import generate_alarm_messages


jjmj_bp = Blueprint('jjmj', __name__)


@jjmj_bp.route('/station/report/<start>/<end>')
def download_report(start, end):
    response = generate_report(start, end)
    return response


@jjmj_bp.route('/station/reduce/data/<start>/<end>')
def download_reduce_data(start, end):
    response = get_reduce_data(start, end)
    return response


@jjmj_bp.route('/station/reduce/picture/<start>/<end>')
def download_reduce_picture(start, end):
    response = get_reduce_picture(start, end)
    return response


@jjmj_bp.route('/station/on')
def get_on_grid_energy():
    df = test_station()
    return jsonify(df)


@jjmj_bp.route('/station/alarm/<start>/<end>')
def generate_alarm_message(start, end):
    response = generate_alarm_messages(start, end)
    return response

