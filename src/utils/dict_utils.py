import ast
import json
import re


def robust_dict_from_str(s: str) -> dict:
    """
    从包含字典字符串的文本中提取并转换字典，支持多种格式和冗余文本

    修复问题：
    1. 预处理布尔值/空值替换时的KeyError
    2. 改进无引号键的处理
    3. 优化括号匹配算法
    4. 增强复杂嵌套处理
    """

    # 预处理：标准化布尔值和null（修复KeyError）
    def replace_bool_null(match):
        word = match.group(0).lower()
        return {"true": "True", "false": "False", "null": "None"}.get(word, word)

    normalized_s = re.sub(
        r"\b(true|false|null)\b", replace_bool_null, s, flags=re.IGNORECASE
    )

    # 尝试1: 直接解析整个字符串为字典
    try:
        return json.loads(normalized_s)
    except json.JSONDecodeError:
        pass

    # 尝试2: 使用literal_eval处理Python风格字典
    try:
        result = ast.literal_eval(normalized_s)
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError):
        pass

    # 尝试3: 提取JSON代码块 (```json...```)
    json_blocks = re.findall(r"```(?:json)?\s*({.*?})\s*```", normalized_s, re.DOTALL)
    if json_blocks:
        last_json_block = json_blocks[-1]  # 获取最后一个匹配项
        try:
            return json.loads(last_json_block)  # 解析最后一个 JSON 块
        except json.JSONDecodeError:
            pass  # 解析失败时忽略

    # 尝试4: 改进的括号匹配提取字典结构
    stack = []
    start_index = -1
    candidates = []  # 存储所有候选字典

    for i, char in enumerate(normalized_s):
        if char == "{":
            if not stack:  # 新的字典开始
                start_index = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_index != -1:  # 匹配到最外层右括号
                    candidate = normalized_s[start_index : i + 1]
                    candidates.append(candidate)
                    start_index = -1  # 重置开始索引
            # 栈空时遇到右括号说明不匹配，忽略

    # 尝试解析所有候选字典（从长到短排序，优先尝试更完整的结构）
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        # 尝试直接解析
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, dict):
                return result
        except (SyntaxError, ValueError):
            pass

        # 尝试修复Python风格的无引号键
        try:
            # 将 {key: value} 转换为 {"key": value}
            fixed_candidate = re.sub(
                r"([{,\s])([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)",
                lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
                candidate,
            )
            result = json.loads(fixed_candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 所有尝试失败
    print(f"Warning: Failed to extract dictionary from string:\n{s[:200]}...")
    return {}
