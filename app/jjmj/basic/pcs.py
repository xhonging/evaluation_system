import pandas as pd
import numpy as np
from datetime import timedelta

from app.common.function import get_url_data, set_index, set_columns
from ..common.function import get_max_count
from ..config import parameters as pt
from .battery import BatteryCluster

from matplotlib.font_manager import FontProperties
from pandas.plotting import register_matplotlib_converters

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
register_matplotlib_converters()


class EnergyStorageUnit(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.clusters = pt.battery['modules']
        self.data = self.get_data()
        self.ac_chg = None
        self.ac_dis = None
        self.dc_chg = None
        self.dc_dis = None

    def get_data(self):
        """
        获取原始PCS数据
        :return: PCS原始数据
        """
        ids = [["analoginput", "{}{}_日充电量".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_日放电量".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_电池组总电压".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_电池组总电流".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_有功功率".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_功率因数".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_放电电能".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_机内温度".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_电池总SOC".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_视在功率".format(pt.prefix, self.name)],
               ["analoginput", "{}{}_电池组状态".format(pt.prefix, self.name)]
               ]

        df = get_url_data(pt.url, ids, pt.func, self.start, self.end, pt.interval)
        df.index = set_index(df)
        df.columns = set_columns(df, pt.prefix)

        return df.dropna()

    def get_ac_chg(self):
        """
        该PCS的日充电量
        :return:以Series形式返回，index：日期，精度到天
        """
        if self.ac_chg:
            return self.ac_chg
        else:
            date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
            df_ac = pd.Series(index=date_range)

            col = ['{}_日充电量'.format(self.name), '{}_有功功率'.format(self.name)]
            df_data = self.data[col]
            for d, df in df_data.groupby(df_data.index.strftime('%Y-%m-%d')):
                df = df.loc[df[col[1]] < -10]
                if df.empty:
                    df_ac[d] = 0
                else:
                    df_ac[d] = df[col[0]][-1]
            self.ac_chg = df_ac.dropna()
            return self.ac_chg

    def get_ac_dis(self):
        """
        该PCS的日放电量
        :return:以Series形式返回，index：日期，精度到天
        """
        if self.ac_dis:
            return self.ac_dis
        else:
            date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
            df_ac = pd.Series(index=date_range)

            col = ['{}_日放电量'.format(self.name), '{}_有功功率'.format(self.name)]
            df_data = self.data[col]
            for d, df in df_data.groupby(df_data.index.strftime('%Y-%m-%d')):
                df = df.loc[df[col[1]] > 1.0]
                if df.empty:
                    df_ac[d] = 0
                else:
                    df_ac[d] = df[col[0]][-1]
                # break
            self.ac_dis = df_ac.dropna()
            return self.ac_dis

    def get_dc_chg_detail(self):
        """
        获取该PCS下所有电池簇具体的每日充电量
        :return: DataFrame格式返回，columns：电池簇名，index：日期，精度到天
        """
        cluster_names = ['{}_{}'.format(self.name, str(i + 1)) for i in range(self.clusters)]
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_dc = pd.DataFrame(columns=cluster_names, index=date_range)
        for name in cluster_names:
            cluster = BatteryCluster(name, self.start, self.end)
            df_dc[name] = cluster.get_chg_detail()

        return df_dc.dropna()

    def get_dc_chg(self):
        """
        计算该PCS下直流侧日充电量
        :return: 以Series返回，index：日期，精度到天
        """
        if self.dc_chg:
            return self.dc_chg
        else:
            df_detail = self.get_dc_chg_detail()
            df_chg = df_detail.sum(axis=1)
            self.dc_chg = df_chg
            return self.dc_chg

    def get_dc_dis_detail(self):
        """
        获取该PCS下所有电池簇具体的每日放电量
        :return: DataFrame格式返回，columns：电池簇名，index：日期，精度到天
        """
        cluster_names = ['{}_{}'.format(self.name, str(i + 1)) for i in range(self.clusters)]
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_dc = pd.DataFrame(columns=cluster_names, index=date_range)
        for name in cluster_names:
            cluster = BatteryCluster(name, self.start, self.end)
            df_dc[name] = cluster.get_dis_detail()

        return df_dc.dropna()

    def get_dc_dis(self):
        """
        计算该PCS下直流侧日放电量
        :return: 以Series返回，index：日期，精度到天
        """
        if self.dc_dis:
            return self.dc_dis
        else:
            df_detail = self.get_dc_dis_detail()
            df_dis = df_detail.sum(axis=1)
            self.dc_dis = df_dis
            return self.dc_dis

    def get_ac_efficiency(self):
        """
        计算PCS交流侧充放电效率
        :return:以Series返回，index：日期，精度到天
        """
        df = pd.DataFrame(columns=['充电电量', '放电电量', 'PCS交流侧效率'])
        df.loc[:, '充电电量'] = self.get_ac_chg()
        df.loc[:, '放电电量'] = self.get_ac_dis()
        df.loc[:, 'PCS交流侧效率'] = df['放电电量'] / df['充电电量'] * 100

        df_efficiency = df['PCS交流侧效率']
        df_efficiency = df_efficiency[df_efficiency < 85]
        df_efficiency = df_efficiency[df_efficiency > 75]
        # print(df_efficiency)

        return df_efficiency

    def get_dc_efficiency(self):
        """
        计算PCS的直流侧充放电效率
        :return:以Series返回，index：日期，精度到天
        """
        df = pd.DataFrame(columns=['充电电量', '放电电量', '直流侧效率'])
        df.loc[:, '充电电量'] = self.get_dc_chg()
        df.loc[:, '放电电量'] = self.get_dc_dis()
        df.loc[:, '直流侧效率'] = df['放电电量'] / df['充电电量'] * 100

        df_efficiency = df['直流侧效率']
        df_efficiency = df_efficiency[df_efficiency < 100]
        df_efficiency = df_efficiency[df_efficiency > 90]
        # print(df_efficiency)

        return df_efficiency

    def get_pcs_chg_efficiency(self):
        """
        计算PCS单侧充电转换效率
        :return:以Series返回，index：日期，精度到天
        """
        df = pd.DataFrame(columns=['直流侧充电电量', '交流侧充电电量', '{}单侧充电效率'.format(self.name)])
        df.loc[:, '直流侧充电电量'] = self.get_dc_chg()
        df.loc[:, '交流侧充电电量'] = self.get_ac_chg()
        df.loc[:, '{}单侧充电效率'.format(self.name)] = df['直流侧充电电量'] / df['交流侧充电电量'] * 100

        df_efficiency = df['{}单侧充电效率'.format(self.name)]
        df_efficiency = df_efficiency[df_efficiency < 100]
        df_efficiency = df_efficiency[df_efficiency > 85]
        # print(df_efficiency)

        return df_efficiency

    def get_pcs_dis_efficiency(self):
        """
        计算PCS单侧放电转换效率
        :return:以Series返回，index：日期，精度到天
        """
        df = pd.DataFrame(columns=['直流侧放电电量', '交流侧放电电量', '{}单侧放电效率'.format(self.name)])
        df.loc[:, '直流侧放电电量'] = self.get_dc_dis()
        df.loc[:, '交流侧放电电量'] = self.get_ac_dis()
        df.loc[:, '{}单侧放电效率'.format(self.name)] = df['交流侧放电电量'] / df['直流侧放电电量'] * 100

        df_efficiency = df['{}单侧放电效率'.format(self.name)]
        df_efficiency = df_efficiency[df_efficiency < 100]
        df_efficiency = df_efficiency[df_efficiency > 85]
        # print(df_efficiency)

        return df_efficiency

    def run(self):
        """
        该类运行函数
        :return: 无
        """
        df = self.get_dc_chg_detail()
        print(df)
