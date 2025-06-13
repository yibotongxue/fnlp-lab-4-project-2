from typing import override

from .base import BaseChargePredictor
from ...llm import BaseLLM, get_llm
from ...utils import robust_dict_from_str


class ZeroShotChargePredictor(BaseChargePredictor):
    system_prompt: str = (
        """你是中华人民共和国的法官，要按照中华人民共和国刑法等法律法规，严格根据检察院指控的犯罪事实，确定被告的罪名，所有判决必须有事实依据，有法律依据，不能主观臆断犯罪情节，也不能脱离法律条文判处刑罚"""
    )

    def __init__(self, llm: str = "qwen-max"):
        self.llm: BaseLLM = get_llm(llm)

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        """
        Predict the charges and sentencing for the given fact and defendants.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.

        Returns:
            dict: A dictionary containing the predicted charges and their details.
        """
        prompt = self.build_zero_shot_prompt(fact, defendants)
        response = self.llm.generate(prompt, system_prompt=self.system_prompt)
        return robust_dict_from_str(response)

    @staticmethod
    def build_zero_shot_prompt(fact: str, defendants: list[str]) -> str:
        """
        Build a zero-shot prompt for the LLM based on the provided fact and defendants.

        Args:
            fact (str): The input fact to include in the prompt.
            defendants (list[str]): A list of defendants to include in the prompt.

        Returns:
            str: The constructed prompt string.
        """
        prompt = f"""你是一个中华人民共和国的法官，你现在需要根据检察院对多名被告的指控，依据中华人民共和国刑法，判断每个被告犯下的所有罪名，并给出每个罪名的量刑，量刑以月为单位。进行判决的时候，你应该注意：
1. 你要完整、仔细的阅读检察官的指控，你的量刑必须严格的按照检察官的指控事实作出，不能脱离事实，也不必怀疑检察官的指控
2. 你应该按照中华人民共和国刑法的法律条款判决，每个被告的罪名都必须在刑法中有所对应，比如“伪证罪”
3. 所有罪名都必须在指控中有明显的直接体现，不能通过主观臆断推定
4. 不同被告可能会有不同的罪名，一个罪行不一定所有被告都会涉及，你需要逐一分析，不能一概而论
5. 你的所有罪名都必须是完整的罪名，不能使用缩写或简称
以下是检察官的指控：
{fact}
以下是被告的名单：
{', '.join(defendants)}
请你根据以上信息，给出每个被告的罪名，在此之前你应该进行详细的分析，对每一个被告要分析指控中他涉及的部分，注意被告可能同时被指控多项罪行，然后分析他可能涉及的罪名，并注意要引用中华人民共和国刑法的法律条文，最后使用JSON格式对之前的分析进行总结，注意只在最后使用JSON格式总结，中间不要使用JSON代码，一个示例如下：
{{"<第一个被告姓名>": ["第一个被告的第一个罪名", "第一个被告的第二个罪名"], "<第二个被告姓名>": ["第二个被告的第一个罪名"]}}
"""
        return prompt
