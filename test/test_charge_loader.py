import pytest

from src.utils.data_utils import ChargeLoader


# 定义 fixture 来初始化 ChargeLoader 实例
@pytest.fixture
def charge_loader():
    return ChargeLoader("./data/charges.json")


def test_all_charges_loaded(charge_loader):
    assert len(list(charge_loader.all_charges.keys())) == 321


def test_load_charge_id(charge_loader):
    assert charge_loader.load_charge_id("伪证罪") == 0
    with pytest.raises(KeyError):
        charge_loader.load_charge_id("不存在罪名")


def test_get_charge_name(charge_loader):
    assert charge_loader.get_charge_name(0) == "伪证罪"
    with pytest.raises(ValueError):
        charge_loader.get_charge_name("0")
