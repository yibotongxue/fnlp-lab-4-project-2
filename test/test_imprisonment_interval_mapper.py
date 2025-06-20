import pytest
from src.utils.imprisonment_mapper import (
    IntervalImprisonmentMapper,
)  # 替换为实际模块路径


class TestIntervalImprisonmentMapper:
    # 初始化参数验证测试
    def test_init_validation(self):
        # 验证空lower_bound报错
        with pytest.raises(
            AssertionError, match="lower bounds should be more than one element"
        ):
            IntervalImprisonmentMapper([], [])

        # 验证首元素非0报错
        with pytest.raises(
            AssertionError, match="the first lower bound should be zero"
        ):
            IntervalImprisonmentMapper([1, 2], [3, 4])

        # 验证长度不等报错
        with pytest.raises(
            AssertionError,
            match="length of lower bound is 2.*length of the represent list is 1",
        ):
            IntervalImprisonmentMapper([0, 1], [5])

        # 验证非递增lower_bound报错
        with pytest.raises(
            AssertionError, match="the lower bound should be incresement"
        ):
            IntervalImprisonmentMapper([0, 2, 1], [0, 2, 3])

        # 验证非递增represent_list报错
        with pytest.raises(
            AssertionError, match="the represent list should be incresement"
        ):
            IntervalImprisonmentMapper([0, 1, 2], [0, 3, 2])

    # imprisonment2label功能测试
    @pytest.fixture
    def mapper(self):
        return IntervalImprisonmentMapper([0, 3, 6, 12, 24], [0, 3, 6, 12, 24])

    @pytest.mark.parametrize(
        "months, expected_label",
        [
            (0, 0),  # 下边界
            (2, 0),  # 第一区间内
            (3, 1),  # 第二区间下边界
            (4, 1),  # 第二区间内
            (5, 1),  # 第二区间内
            (6, 2),  # 第三区间下边界
            (11, 2),  # 第三区间内
            (12, 3),  # 第四区间下边界
            (23, 3),  # 第四区间内
            (24, 4),  # 第五区间下边界
            (100, 4),  # 第五区间内
        ],
    )
    def test_imprisonment2label(self, mapper, months, expected_label):
        assert mapper.imprisonment2label(months) == expected_label

    # imprisonment2label异常输入测试
    def test_imprisonment2label_invalid(self, mapper):
        with pytest.raises(
            AssertionError, match="imprisonment should be not less than zero"
        ):
            mapper.imprisonment2label(-1)

    # label2imprisonment功能测试
    @pytest.mark.parametrize(
        "label, expected_months",
        [
            (0, 0),  # 首标签
            (1, 3),  # 第二标签
            (2, 6),  # 第三标签
            (3, 12),  # 第四标签
            (4, 24),  # 末标签
        ],
    )
    def test_label2imprisonment(self, mapper, label, expected_months):
        assert mapper.label2imprisonment(label) == expected_months

    # label2imprisonment异常输入测试
    @pytest.mark.parametrize("invalid_label", [-1, 5, 100])
    def test_label2imprisonment_invalid(self, mapper, invalid_label):
        with pytest.raises(
            AssertionError,
            match="label should be not less than zero and less than length of represent list",
        ):
            mapper.label2imprisonment(invalid_label)

    # 单区间边界测试
    def test_single_interval(self):
        mapper = IntervalImprisonmentMapper([0], [0])
        assert mapper.imprisonment2label(0) == 0
        assert mapper.imprisonment2label(100) == 0
        assert mapper.label2imprisonment(0) == 0

    # 非连续represent值测试
    def test_non_continuous_represent(self):
        mapper = IntervalImprisonmentMapper([0, 5, 10], [1, 7, 15])
        assert mapper.imprisonment2label(3) == 0  # 0-5区间应返回标签0
        assert mapper.label2imprisonment(0) == 1  # 标签0对应represent值1
        assert mapper.label2imprisonment(1) == 7  # 标签1对应represent值7
