import os
import pandas as pd
from app.common.function import get_url_data, set_index, set_columns
from ..common.function import cal_chg_number
from ..common.function import delete_outliers
from ..config import parameters as pt

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pandas.plotting import register_matplotlib_converters

font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
register_matplotlib_converters()


class BatteryCluster(object):
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """
        获取电池簇采集数据
        :return: 电池簇采集数据
        """
        ids = [['analoginput', pt.prefix + self.name + '#电池组SOC'],
               ['analoginput', pt.prefix + self.name + '#电池组SOH'],
               ['analoginput', pt.prefix + self.name + '#电池组平均温度'],
               ['analoginput', pt.prefix + self.name + '#电池组平均电压'],
               ['analoginput', pt.prefix + self.name + '#电池组总压'],
               ['analoginput', pt.prefix + self.name + '#电池组日充电量'],
               ['analoginput', pt.prefix + self.name + '#电池组日放电量'],
               ['analoginput', pt.prefix + self.name + '#电池组最低温度'],
               ['analoginput', pt.prefix + self.name + '#电池组最低电压'],
               ['analoginput', pt.prefix + self.name + '#电池组最大允许充电电流'],
               ['analoginput', pt.prefix + self.name + '#电池组最大允许放电电流'],
               ['analoginput', pt.prefix + self.name + '#电池组最高温度'],
               ['analoginput', pt.prefix + self.name + '#电池组最高电压'],
               ['analoginput', pt.prefix + self.name + '#电池组状态字'],
               ['analoginput', pt.prefix + self.name + '#电池组电流']]

        df = get_url_data(pt.url, ids, pt.func, self.start, self.end, pt.interval)
        df.index = set_index(df)
        df.columns = set_columns(df, pt.prefix)

        return df.dropna()

    def get_chg_detail(self):
        """
        获取该电池簇具体的每日充电量
        :return: Series格式返回，index：日期，精度到天
        """
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_dc = pd.Series(index=date_range)
        cluster = BatteryCluster(self.name, self.start, self.end)
        col = ['{}#电池组日充电量'.format(self.name), '{}#电池组电流'.format(self.name)]
        df_data = cluster.data[col]
        for d, df in df_data.groupby(df_data.index.strftime('%Y-%m-%d')):
            df = df.loc[df[col[1]] < 0]
            df = df.loc[df[col[1]] > -50]
            if df.empty:
                df_dc[d] = 0
            else:
                df_dc[d] = df[col[0]][-1]

        return df_dc.dropna()

    def get_dis_detail(self):
        """
        获取该电池簇具体的每日放电量
        :return: Series格式返回，index：日期，精度到天
        """
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_dc = pd.Series(index=date_range)
        cluster = BatteryCluster(self.name, self.start, self.end)
        col = ['{}#电池组日充电量'.format(self.name), '{}#电池组电流'.format(self.name)]
        df_data = cluster.data[col]
        for d, df in df_data.groupby(df_data.index.strftime('%Y-%m-%d')):
            df = df.loc[df[col[1]] > 0]
            df = df.loc[df[col[1]] < 50]
            if df.empty:
                df_dc[d] = 0
            else:
                df_dc[d] = df[col[0]][-1]

        return df_dc.dropna()

    def get_chg_soc(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_soc = pd.DataFrame(columns=['max', 'min'], index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_dis = data['{}#电池组SOC'.format(self.name)]
        for d, df in df_dis.groupby(df_dis.index.strftime('%Y-%m-%d')):
            df_soc['max'][d] = df.max()
            df_soc['min'][d] = df.min()

        return df_soc

    def get_chg_vol(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_vol = pd.DataFrame(columns=['max', 'min'], index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_dis = data[['{}#电池组最高电压'.format(self.name), '{}#电池组最低电压'.format(self.name)]]
        for d, df in df_dis.groupby(df_dis.index.strftime('%Y-%m-%d')):
            df_vol['max'][d] = df['{}#电池组最高电压'.format(self.name)].max()
            df_vol['min'][d] = df['{}#电池组最低电压'.format(self.name)].min()

        return df_vol

    def get_chg_temp(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_temp = pd.DataFrame(columns=['max', 'min'], index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_dis = data[['{}#电池组最高温度'.format(self.name), '{}#电池组最低温度'.format(self.name)]]
        for d, df in df_dis.groupby(df_dis.index.strftime('%Y-%m-%d')):
            df_temp['max'][d] = df['{}#电池组最高温度'.format(self.name)].max()
            df_temp['min'][d] = df['{}#电池组最低温度'.format(self.name)].min()

        return df_temp

    def get_chg_span(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_span = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_chg = data['{}#电池组电流'.format(self.name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            df_span[d] = (df.index[-1] - df.index[0]).total_seconds()

        return df_span

    def get_chg_elec(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_elec = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_chg = data['{}#电池组电流'.format(self.name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            df_elec[d] = abs(df.mean())

        return df_elec

    def get_chg_diff_soc(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_diff_soc = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_chg = data['{}#电池组SOC'.format(self.name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            df_diff_soc[d] = df[-1] - df[0]
        # df_soc = self.get_chg_soc()
        # df_diff_soc = df_soc['max'] - df_soc['min']

        return df_diff_soc

    def get_chg_mean_vol(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_mean_vol = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_chg = data['{}#电池组平均电压'.format(self.name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            df_mean_vol[d] = df.mean()

        return df_mean_vol

    def get_chg_mean_temp(self):
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_mean_temp = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]
        df_chg = data['{}#电池组平均温度'.format(self.name)]
        for d, df in df_chg.groupby(df_chg.index.strftime('%Y-%m-%d')):
            df_mean_temp[d] = df.mean()

        return df_mean_temp

    def get_chg_number(self):
        """
        获取累计充电次数
        :return:
        """
        dir_ = os.path.join(os.getcwd(), 'excel', '累计充放电次数.xlsx')
        if os.path.exists(dir_):
            df = pd.read_excel(dir_, index_col=0)
        else:
            df = cal_chg_number()
        df_chg_number = df[self.name]

        return df_chg_number

    def get_chg_power(self):
        """
        获取充电时刻功率
        :return:
        """
        date_range = [d for d in pd.date_range(self.start, self.end, freq='d', closed='left')]
        df_power = pd.Series(index=date_range)
        data = self.data.loc[self.data['{}#电池组电流'.format(self.name)] < 0.0]

        for d, df in data.groupby(data.index.strftime('%Y-%m-%d')):
            df_power[d] = df['{}#电池组总压'.format(self.name)]*df['{}#电池组电流'.format(self.name)]

        return df_power

    def get_temp_diff(self):
        """
        获取电池组瞬时温度极差数据
        :return: 电池组瞬时温度极差数据
        """
        data = self.data[[self.name + '#电池组最高温度',
                        self.name + '#电池组最低温度',
                        self.name + '#电池组平均温度']]
        print(data)
        df_temp_diff = data.diff()
        date_range = pd.date_range(self.start, self.end, freq='d', closed='left')
        max_temp_diff = pd.DataFrame(columns=data.columns, index=date_range)
        for d, df in df_temp_diff.groupby(df_temp_diff.index.strftime('%Y-%m-%d')):
            max_temp_diff.loc[d] = abs(df).max()
            # print(max_temp_diff)

        return data

    def get_max_temp(self):
        """
        获取电池组瞬时温度极差数据
        :return: 电池组瞬时温度极差数据
        """
        data = self.data[[self.name + '#电池组最高温度']]
        date_range = pd.date_range(self.start, self.end, freq='d', closed='left')
        max_temp = pd.DataFrame(columns=data.columns, index=date_range)
        for d, df in data.groupby(data.index.strftime('%Y-%m-%d')):
            max_temp.loc[d] = abs(df).max()
        return max_temp

    def run(self):
        df = self.data
        print(df)

    def get_chg_vol_max_min(self):
        """
        获取电池模块充电时段电压极差数据
        :return: 电池模块充电时段电压极差数据
        """
        df = self.data.loc[self.data[self.name + '#电池组电流'] < 0.0]
        vol_data = 1000*(df[self.name + '#电池组最高电压'] - df[self.name + '#电池组最低电压'])

        return vol_data

    def get_temp_max_min(self):
        """
        获取电池组充电时段温度极差数据
        :return: 电池组充电时段温度极差数据
        """
        df = self.data.loc[self.data[self.name + '#电池组电流'] < 0.0]
        temp_data = self.data[self.name + '#电池组最高温度'] - self.data[self.name + '#电池组最低温度']

        return temp_data





