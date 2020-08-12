import json
import pandas as pd
import numpy as np
from urllib import parse, request
from datetime import datetime


def get_url_data(url_base, ids, func, begin, end, itv):
    """
    从云平台获取数据
    :param url_base:云平台地址
    :param ids:数据表头
    :param func:如何取值，如max，min等
    :param begin:开始时间
    :param end:结束时间
    :param itv:时间间隔
    :return:数据
    """
    url = url_base + "/report/complex?"
    parameters = {}
    parameters["captions"] = "{" + ",".join(
        [x[0] + ":" + x[1] + ":" + x[1] if len(x) < 3 else x[0] + ":" + x[1] + ":" + x[2] + ":" + x[2] for x in
         ids]) + "}"
    parameters["interval"] = itv

    parameters["begin"] = begin
    if end != "now":
        parameters["end"] = end
    parameters["func"] = func
    url = url + parse.urlencode(parameters)
    # print(url)
    try:
        strHtml = request.urlopen(url, timeout=10.0).read()
        ret = json.loads(strHtml.decode("utf-8"))
        return pd.DataFrame(ret, dtype=float)
    except Exception as e:
        raise ConnectionError('连接错误！请检查url！错误信息: {}'.format(e))


def set_index(df):
    """
    转换Dataframe的Index由时间戳进行转化成DatetimeIndex
    :param df: 待转Dataframe
    :return: DatetimeIndex
    """
    df.index = pd.to_datetime(df.index, utc=True, unit='s').tz_convert("Asia/Shanghai").strftime(
        "%Y-%m-%d %H:%M:%S")
    return pd.DatetimeIndex(df.index)


def set_columns(df, prefix):
    """
    简化Dataframe的列名称
    :param df: Dataframe
    :param prefix: 待简化部分字段
    :return: 列名简化后Dataframe
    """
    columns = df.columns.map(lambda x: x.replace(prefix, ''))
    return columns


def get_valid_date(str_):
    """
    判断是否是一个有效的日期字符串
    :param str_: 待判断字符串
    :return: datetime格式日期
    """
    try:
        if ":" in str_:
            return datetime.strptime(str_, "%Y-%m-%d %H:%M:%S")
        else:
            return datetime.strptime(str_, "%Y-%m-%d")
    except ValueError:
        raise ValueError('时间格式错误！正确格式示例：1970-01-01')



