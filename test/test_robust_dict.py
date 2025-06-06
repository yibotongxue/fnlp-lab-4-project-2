from src.utils import robust_dict_from_str


# 基础测试用例
def test_valid_json():
    s = '{"name": "Alice", "age": 30, "active": true}'
    result = robust_dict_from_str(s)
    assert result == {"name": "Alice", "age": 30, "active": True}


def test_python_style_dict():
    s = "{'name': 'Bob', 'scores': [90, 85], 'active': False}"
    result = robust_dict_from_str(s)
    assert result == {"name": "Bob", "scores": [90, 85], "active": False}


def test_mixed_quotes():
    s = """{'key1': "value1", "key2": 'value2'}"""
    result = robust_dict_from_str(s)
    assert result == {"key1": "value1", "key2": "value2"}


# JSON代码块测试
def test_json_code_block():
    s = """Here is the response:
    ```json
    {
        "status": "success",
        "code": 200
    }
    ```
    Thank you!"""
    result = robust_dict_from_str(s)
    assert result == {"status": "success", "code": 200}


def test_json_code_block_no_label():
    s = "Prefix ```\n{'result': 'ok'}\n``` Suffix"
    result = robust_dict_from_str(s)
    assert result == {"result": "ok"}


# 嵌套结构测试
def test_nested_dict():
    s = "Output: {'user': {'name': 'Charlie', 'roles': ['admin', 'user']}} End"
    result = robust_dict_from_str(s)
    assert result == {"user": {"name": "Charlie", "roles": ["admin", "user"]}}


def test_complex_nesting():
    s = "Data: {'a': 1, 'b': {'c': [2, 3], 'd': {'e': 4}}}"
    result = robust_dict_from_str(s)
    assert result == {"a": 1, "b": {"c": [2, 3], "d": {"e": 4}}}


# 冗余文本处理
def test_prefix_text():
    s = "The answer is: {'key': 'value'}"
    result = robust_dict_from_str(s)
    assert result == {"key": "value"}


def test_suffix_text():
    s = "{'temperature': 25.5} Celsius"
    result = robust_dict_from_str(s)
    assert result == {"temperature": 25.5}


def test_multiple_dicts():
    s = "First: {'a': 1} Second: {'b': 2} Third: {'c': 3}"
    result = robust_dict_from_str(s)
    # 应该提取第一个完整的字典
    assert result == {"a": 1}


# 特殊字符和格式
def test_special_characters():
    s = """{'message': "Hello\\nWorld!", "path": "C:\\\\Windows"}"""
    result = robust_dict_from_str(s)
    assert result == {"message": "Hello\nWorld!", "path": "C:\\Windows"}


def test_single_quoted_keys():
    s = "{'first-name': 'John', 'last-name': 'Doe'}"
    result = robust_dict_from_str(s)
    assert result == {"first-name": "John", "last-name": "Doe"}


def test_trailing_comma():
    s = '{"items": [1, 2, 3,],}'
    result = robust_dict_from_str(s)
    # 应该能容忍JSON不允许的尾随逗号
    assert result == {"items": [1, 2, 3]}


# 无效情况处理
def test_invalid_dict():
    s = "{'key': 'value' 'missing_comma': true}"
    result = robust_dict_from_str(s)
    assert result == {}


def test_no_dict():
    s = "This is just a regular string"
    result = robust_dict_from_str(s)
    assert result == {}


def test_unclosed_brace():
    s = "Partial: {'key': 'value"
    result = robust_dict_from_str(s)
    assert result == {}


# 真实LLM响应模拟
def test_realistic_llm_response():
    s = """
    Sure! Here's the JSON response you requested:

    ```json
    {
        "id": 12345,
        "name": "Example",
        "tags": ["test", "demo"],
        "metadata": {
            "created": "2023-01-01",
            "modified": "2023-05-15"
        }
    }
    ```

    Let me know if you need anything else!
    """
    result = robust_dict_from_str(s)
    assert result == {
        "id": 12345,
        "name": "Example",
        "tags": ["test", "demo"],
        "metadata": {"created": "2023-01-01", "modified": "2023-05-15"},
    }


def test_llm_response_with_extra_braces():
    s = "Response: { 'outer': { 'inner': {'value': 42} } } (End of response)"
    result = robust_dict_from_str(s)
    assert result == {"outer": {"inner": {"value": 42}}}


# 空值和布尔值处理
def test_null_and_boolean():
    s = "{'is_valid': false, 'error': null}"
    result = robust_dict_from_str(s)
    assert result == {"is_valid": False, "error": None}


# 性能测试
def test_large_dict():
    large_dict = "{"
    for i in range(1000):
        large_dict += f'"{i}": {i},'
    large_dict = large_dict.rstrip(",") + "}"

    s = f"Large data: {large_dict}"
    result = robust_dict_from_str(s)
    assert len(result) == 1000
    assert result["999"] == 999


# 测试多个候选字典
def test_multiple_dict_candidates():
    s = "First {a:1} Second {b:2} Third {c:3}"
    result = robust_dict_from_str(s)
    # 应该提取第一个完整的字典
    assert result == {"a": 1}
