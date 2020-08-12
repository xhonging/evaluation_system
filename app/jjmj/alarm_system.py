# @Time    : 2020-08-06 9:11
# @Author  : 洪星
# @File    : alarm_system.py

import os
import pandas as pd
import numpy as np
from ..common.function import get_url_data, set_index, set_columns
from .config import parameters as pt


def generate_alarm_messages(start, end):
    """
    数据异常即时报警系统
    :param start: 数据采集开始时间
    :param end: 数据采集结束时间
    :return:
    """
    # 电池安全报警
    battery_alarm(start, end)

    # 环境温度报警
    environment_temperature_alarm(start, end)

    # 系统安全报警
    system_alarm(start, end)


def communication_alarm(df):
    """
    通讯故障报警
    :param df: 获取的通信数据
    :return: 报警信息
    """
    if df.empty:
        return '暂无法获取数据，请排查原因！'
    else:
        return None


def battery_alarm(start, end):
    """
    电池安全报警
    :param start: 开始时间
    :param end: 结束时间
    :return: 报警信息
    """
    battery_alarm_sub_1(start, end)

    battery_alarm_sub_2(start, end)


def battery_alarm_sub_1(start, end):
    pass


def battery_alarm_sub_2(start, end):
    pass


def environment_temperature_alarm(start, end):
    """
    空调环境温度报警
    :param start: 开始时间
    :param end: 结束时间
    :return: 报警信息
    """
    ids = [['analoginput', pt.prefix + '空调1_室内温度'],
           ['analoginput', pt.prefix + '空调2_室内温度'],
           ['analoginput', pt.prefix + '空调3_室内温度'],
           ['analoginput', pt.prefix + '空调4_室内温度']]

    df = get_url_data(pt.url, ids, pt.func, start, end, '1m')
    df.index = set_index(df)
    df.columns = set_columns(df, pt.prefix)

    mean_temp = df.mean(axis=1)

    if mean_temp > 32:
        return '警告！室内温度过高，已达{}℃，请迅速处理！'.format(mean_temp)
    else:
        return None


def system_alarm(start, end):
    pass

