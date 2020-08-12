from .basic.battery import BatteryCluster
from .config import parameters as pt
from .basic.pcs import EnergyStorageUnit
from .indicator import vol_max_min_all, temp_max_min_all, soc_max_min_all, eva_sys
from .basic.station import Station


def test_1():
    name = 'PCS1_1'
    PCS1_1 = BatteryCluster(name, pt.start, pt.end)
    PCS1_1.run()


def test_2():
    PCS1 = EnergyStorageUnit('PCS1', pt.start, pt.end)
    PCS1.run()


def test_3():
    for i in range(pt.battery['PCS']):
        name = 'PCS' + str(i+1)
        PCS = EnergyStorageUnit(name, pt.start, pt.end)
        PCS.run()
        # break


def test_4():
    # vol_max_min_all(1)
    # vol_max_min_all(2)
    vol_max_min_all(3)


def test_5():
    # temp_max_min_all(1)
    # temp_max_min_all(2)
    temp_max_min_all(3)


def test_6():
    # soc_max_min_all(1)
    # soc_max_min_all(2)
    soc_max_min_all(3)


def test_7():
    eva_sys()


def test_station():
    station = Station(pt.project, pt.start, pt.end)
    df = station.get_on_grid_energy()
    return df

