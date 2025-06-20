import re
from typing import override

from .base import BaseRefiner
from ....llm import get_llm


class ZeroShotRefiner(BaseRefiner):
    def __init__(self, candidate_cnt: int, llm: str):
        super().__init__(candidate_cnt)
        self.llm = get_llm(llm)

    @override
    def refine(self, fact: str, defendant: str, candidate: list[str]) -> str:
        prompt = self.build_prompt(fact, defendant, candidate)
        response = self.llm.generate(prompt)
        return [self.extract_answer(response)]

    def build_prompt(self, fact: str, defendant: str, candidate: list[str]) -> str:
        prompt = f"""你是一个北京大学法律专业的学生，十分熟悉中华人民共和国法律特别是刑法，现在你需要根据中华人民共和国刑法的有关规定，对于用户给出的指控事实和被告，给出被告的罪名。
用户会提供{self.candidate_cnt}个候选项，你需要首先分析用户给出的指控事实，特别关注给定被告在指控事实中犯下的罪行，然后逐个核对用户给出的{self.candidate_cnt}个候选罪名，分析其是否是该用户的合理罪名，然后给出最终的罪名。
注意最终的罪名必须来自用户提供的候选项，不能修改，并用<answer></answer>标签包围最终的罪名，比如<answer>伪证罪</answer>
以下是犯罪指控事实：
{fact}
以下是要确定罪名的被告：
{defendant}
以下是所有候选的可能罪名：
{candidate}
请按照上面的流程分析并给出最终答案。
"""
        return prompt

    @staticmethod
    def extract_answer(response: str) -> str:
        pattern = re.compile(r"<answer>([^<]*)</answer>")
        matcher = pattern.findall(response)
        if len(matcher) < 1:
            raise ValueError(f'No answer found in "{response}"')
        return matcher[-1]
