from typing import override

from ..llm import BaseLLM
from ..utils import robust_dict_from_str
from ..utils.type_utils import OutcomeDict
from .base import BasePredictor


class ZeroShotPredictor(BasePredictor):
    """Zero-shot predictor that uses an LLM to generate responses based on prompts."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @override
    def predict_judgment(self, fact: str, defendants: list[str]) -> list[OutcomeDict]:
        """
        Generate a response based on the provided fact using the LLM.

        Args:
            fact (str): The input fact to generate a response for.
            defendants (list[str]): A list of defendants to include in the response.

        Returns:
            list[OutcomeDict]: A list of predicted outcomes, each containing the name and judgment details.
        """
        prompt = self.build_zero_shot_prompt(fact, defendants)
        response = self.llm.generate(
            prompt,
        )
        outcomes = robust_dict_from_str(response)
        assert "outcomes" in outcomes, "The response does not contain 'outcomes' key."
        assert isinstance(outcomes["outcomes"], list), "'outcomes' should be a list."
        result = []
        for outcome in outcomes["outcomes"]:
            try:
                result.append(OutcomeDict(**outcome))
            except:
                raise ValueError(f"Invalid outcome format: {outcome}")
        return result

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
        # TODO Implement the actual prompt construction logic
        prompt = f"""
        你是一个中华人民共和国的法官，你现在需要根据检察院对多名被告的指控，依据中华人民共和国刑法，
        判断每个被告犯下的所有罪名，并给出每个罪名的量刑，量刑以月为单位。进行判决的时候，你应该注意：
        1. 你要完整、仔细的阅读检察官的指控，你的量刑必须严格的按照检察官的指控事实作出，不能脱离事实，也不必怀疑检察官的指控
        2. 你应该按照中华人民共和国刑法的法律条款判决，每个被告的罪名都必须在刑法中有所对应，比如“伪证罪”
        3. 所有罪名都必须在指控中有明显的直接体现，不能通过主观臆断推定
        4. 每个罪名的量刑必须在中华人民共和国刑法的规定范围内，必须严格的按照法律条款来量刑
        5. 你需要给出每个罪名的量刑，量刑以月为单位，量刑需要是整数，注意你需要给出完整的判决，包括有期徒刑、罚金等等，但最终的总结中你只需要给出有期徒刑的量刑，其他的刑罚不必考虑，如果不必判处有期徒刑（这是很可能的），则量刑为0
        6. 特别注意被告犯下罪行的时间，如果对应的法律条款后续有修改，要遵从从旧兼从轻的原则
        以下是检察官的指控：
        \n{fact}\n
        以下是被告的名单：
        {', '.join(defendants)}
        请你根据以上信息，给出每个被告的罪名和量刑，在此之前你应该进行详细的分析，对每一个被告要分析指控中他涉及的部分，注意被告可能同时被指控多项罪行，然后分析他可能涉及的罪名，并注意要引用中华人民共和国刑法的法律条文，然后对他的每一条成立的罪名根据罪行和法律条文分析量刑，
        最后使用JSON格式对之前的分析进行总结，一个示例如下：
        {{"outcomes": [{{"name": "第一个被告姓名", "judgment": [{{"standard_accusation": "第一个被告第一个罪名", "imprisonment": <对应的刑期，使用整数>}}, {{"standard_accusation": "第一个被告的第二个罪名", "imprisonment": <对应的刑期，使用整数>}}]}}, {{"name": "第二个被告", "judgment": [{{"standard_accusation": "第二个被告的第一个罪名", "imprisonment": "<对应的刑期，使用整数>"}}]}}]}}
        """
        return prompt
