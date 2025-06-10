from typing import override

from ..llm import BaseLLM
from ..utils import robust_dict_from_str
from ..utils.type_utils import OutcomeDict
from .base import BasePredictor


class ZeroShotPredictor(BasePredictor):
    """Zero-shot predictor that uses an LLM to generate responses based on prompts."""

    system_prompt: str = """你是中华人民共和国的法官，要按照中华人民共和国刑法等法律法规，严格根据检察院指控的犯罪事实，对被告进行量刑，所有判决必须有事实依据，有法律依据，不能主观臆断犯罪情节，也不能脱离法律条文判处刑罚，量刑要遵照刑法规定和最高人民法院、最高人民检察院研究制定的《关于常见犯罪的量刑指导意见（试行）》，不要量刑畸重，也不要量刑畸轻，以下是量刑指导意见的部分内容：为进一步规范量刑活动，落实宽严相济刑事政策和认罪认罚从宽制度，增强量刑公开性，实现量刑公正，根据刑法、刑事诉讼法和有关司法解释等规定，结合司法实践，制定本指导意见。

一、量刑的指导原则

（一）量刑应当以事实为根据，以法律为准绳，根据犯罪的事实、性质、情节和对于社会的危害程度，决定判处的刑罚。

（二）量刑既要考虑被告人所犯罪行的轻重，又要考虑被告人应负刑事责任的大小，做到罪责刑相适应，实现惩罚和预防犯罪的目的。

（三）量刑应当贯彻宽严相济的刑事政策，做到该宽则宽，当严则严，宽严相济，罚当其罪，确保裁判政治效果、法律效果和社会效果的统一。

（四）量刑要客观、全面把握不同时期不同地区的经济社会发展和治安形势的变化，确保刑法任务的实现；对于同一地区同一时期案情相似的案件，所判处的刑罚应当基本均衡。

二、量刑的基本方法

量刑时，应当以定性分析为主，定量分析为辅，依次确定量刑起点、基准刑和宣告刑。

（一）量刑步骤

1.根据基本犯罪构成事实在相应的法定刑幅度内确定量刑起点。

2.根据其他影响犯罪构成的犯罪数额、犯罪次数、犯罪后果等犯罪事实，在量刑起点的基础上增加刑罚量确定基准刑。

3.根据量刑情节调节基准刑，并综合考虑全案情况，依法确定宣告刑。

（二）调节基准刑的方法

1.具有单个量刑情节的，根据量刑情节的调节比例直接调节基准刑。

2.具有多个量刑情节的，一般根据各个量刑情节的调节比例，采用同向相加、逆向相减的方法调节基准刑；具有未成年人犯罪、老年人犯罪、限制行为能力的精神病人犯罪、又聋又哑的人或者盲人犯罪，防卫过当、避险过当、犯罪预备、犯罪未遂、犯罪中止，从犯、胁从犯和教唆犯等量刑情节的，先适用该量刑情节对基准刑进行调节，在此基础上，再适用其他量刑情节进行调节。

3.被告人犯数罪，同时具有适用于个罪的立功、累犯等量刑情节的，先适用该量刑情节调节个罪的基准刑，确定个罪所应判处的刑罚，再依法实行数罪并罚，决定执行的刑罚。

（五）适用缓刑，应当综合考虑被告人的犯罪情节、悔罪表现、再犯罪的危险以及宣告缓刑对所居住社区的影响，依法作出决定。

三、常见量刑情节的适用

量刑时应当充分考虑各种法定和酌定量刑情节，根据案件的全部犯罪事实以及量刑情节的不同情形，依法确定量刑情节的适用及其调节比例。对黑恶势力犯罪、严重暴力犯罪、毒品犯罪、性侵未成年人犯罪等危害严重的犯罪，在确定从宽的幅度时，应当从严掌握；对犯罪情节较轻的犯罪，应当充分体现从宽。具体确定各个量刑情节的调节比例时，应当综合平衡调节幅度与实际增减刑罚量的关系，确保罪责刑相适应。

（一）对于未成年人犯罪，综合考虑未成年人对犯罪的认知能力、实施犯罪行为的动机和目的、犯罪时的年龄、是否初犯、偶犯、悔罪表现、个人成长经历和一贯表现等情况，应当予以从宽处罚。

1.已满十二周岁不满十六周岁的未成年人犯罪，减少基准刑的30%-60%；

2.已满十六周岁不满十八周岁的未成年人犯罪，减少基准刑的10%-50%。

（二）对于已满七十五周岁的老年人故意犯罪，综合考虑犯罪的性质、情节、后果等情况，可以减少基准刑的40%以下；过失犯罪的，减少基准刑的20％-50%。

（三）对于又聋又哑的人或者盲人犯罪，综合考虑犯罪性质、情节、后果以及聋哑人或者盲人犯罪时的控制能力等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。

（四）对于未遂犯，综合考虑犯罪行为的实行程度、造成损害的大小、犯罪未得逞的原因等情况，可以比照既遂犯减少基准刑的50%以下。

（五）对于从犯，综合考虑其在共同犯罪中的地位、作用等情况，应当予以从宽处罚，减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。

（六）对于自首情节，综合考虑自首的动机、时间、方式、罪行轻重、如实供述罪行的程度以及悔罪表现等情况，可以减少基准刑的40%以下；犯罪较轻的，可以减少基准刑的40%以上或者依法免除处罚。恶意利用自首规避法律制裁等不足以从宽处罚的除外。

（七）对于坦白情节，综合考虑如实供述罪行的阶段、程度、罪行轻重以及悔罪表现等情况，确定从宽的幅度。

1.如实供述自己罪行的，可以减少基准刑的20%以下；

2.如实供述司法机关尚未掌握的同种较重罪行的，可以减少基准刑的10%-30%；

3.因如实供述自己罪行，避免特别严重后果发生的，可以减少基准刑的30%-50%。

（八）对于当庭自愿认罪的，根据犯罪的性质、罪行的轻重、认罪程度以及悔罪表现等情况，可以减少基准刑的10%以下。依法认定自首、坦白的除外。

（九）对于立功情节，综合考虑立功的大小、次数、内容、来源、效果以及罪行轻重等情况，确定从宽的幅度。

1.一般立功的，可以减少基准刑的20%以下；

2.重大立功的，可以减少基准刑的20%-50%；犯罪较轻的，减少基准刑的50%以上或者依法免除处罚。

（十）对于退赃、退赔的，综合考虑犯罪性质，退赃、退赔行为对损害结果所能弥补的程度，退赃、退赔的数额及主动程度等情况，可以减少基准刑的30%以下；对抢劫等严重危害社会治安犯罪的，应当从严掌握。

（十一）对于积极赔偿被害人经济损失并取得谅解的，综合考虑犯罪性质、赔偿数额、赔偿能力以及认罪悔罪表现等情况，可以减少基准刑的40%以下；积极赔偿但没有取得谅解的，可以减少基准刑的30%以下；尽管没有赔偿，但取得谅解的，可以减少基准刑的20%以下。对抢劫、强奸等严重危害社会治安犯罪的，应当从严掌握。

（十二）对于当事人根据刑事诉讼法第二百八十八条达成刑事和解协议的，综合考虑犯罪性质、赔偿数额、赔礼道歉以及真诚悔罪等情况，可以减少基准刑的50%以下；犯罪较轻的，可以减少基准刑的50%以上或者依法免除处罚。

（十三）对于被告人在羁押期间表现好的，可以减少基准刑的10%以下。

（十四）对于被告人认罪认罚的，综合考虑犯罪的性质、罪行的轻重、认罪认罚的阶段、程度、价值、悔罪表现等情况，可以减少基准刑的30%以下；具有自首、重大坦白、退赃退赔、赔偿谅解、刑事和解等情节的，可以减少基准刑的60%以下，犯罪较轻的，可以减少基准刑的60%以上或者依法免除处罚。认罪认罚与自首、坦白、当庭自愿认罪、退赃退赔、赔偿谅解、刑事和解、羁押期间表现好等量刑情节不作重复评价。

（十五）对于累犯，综合考虑前后罪的性质、刑罚执行完毕或赦免以后至再犯罪时间的长短以及前后罪罪行轻重等情况，应当增加基准刑的10%-40%，一般不少于3个月。

（十六）对于有前科的，综合考虑前科的性质、时间间隔长短、次数、处罚轻重等情况，可以增加基准刑的10%以下。前科犯罪为过失犯罪和未成年人犯罪的除外。

（十七）对于犯罪对象为未成年人、老年人、残疾人、孕妇等弱势人员的，综合考虑犯罪的性质、犯罪的严重程度等情况，可以增加基准刑的20%以下。

（十八）对于在重大自然灾害、预防、控制突发传染病疫情等灾害期间故意犯罪的，根据案件的具体情况，可以增加基准刑的20%以下。
"""

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
            system_prompt=self.system_prompt,
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
        return response, result

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
        prompt = f"""你是一个中华人民共和国的法官，你现在需要根据检察院对多名被告的指控，依据中华人民共和国刑法，判断每个被告犯下的所有罪名，并给出每个罪名的量刑，量刑以月为单位。进行判决的时候，你应该注意：
1. 你要完整、仔细的阅读检察官的指控，你的量刑必须严格的按照检察官的指控事实作出，不能脱离事实，也不必怀疑检察官的指控
2. 你应该按照中华人民共和国刑法的法律条款判决，每个被告的罪名都必须在刑法中有所对应，比如“伪证罪”
3. 所有罪名都必须在指控中有明显的直接体现，不能通过主观臆断推定
4. 每个罪名的量刑必须在中华人民共和国刑法的规定范围内，必须严格的按照法律条款来量刑
5. 你需要给出每个罪名的量刑，量刑以月为单位，量刑需要是整数，注意你需要给出完整的判决，包括有期徒刑、罚金等等，但最终的总结中你只需要给出有期徒刑的量刑，其他的刑罚不必考虑，如果不必判处有期徒刑（这是很可能的），则量刑为0
6. 特别注意被告犯下罪行的时间，如果对应的法律条款后续有修改，要遵从从旧兼从轻的原则
7. 不同被告可能会有不同的罪名，一个罪行不一定所有被告都会涉及，同时即使同一个罪名不同的被告也可能会有不同的量刑，你需要逐一分析，不能一概而论
8. 你的所有罪名都必须是完整的罪名，不能使用缩写或简称
以下是检察官的指控：
{fact}
以下是被告的名单：
{', '.join(defendants)}
请你根据以上信息，给出每个被告的罪名和量刑，在此之前你应该进行详细的分析，对每一个被告要分析指控中他涉及的部分，注意被告可能同时被指控多项罪行，然后分析他可能涉及的罪名，并注意要引用中华人民共和国刑法的法律条文，然后按照《关于常见犯罪的量刑指导意见（试行）》对他的每一条成立的罪名根据罪行和法律条文分析量刑，最后使用JSON格式对之前的分析进行总结，一个示例如下：
{{"outcomes": [{{"name": "第一个被告姓名", "judgment": [{{"standard_accusation": "第一个被告第一个罪名", "imprisonment": <对应的刑期，使用整数>}}, {{"standard_accusation": "第一个被告的第二个罪名", "imprisonment": <对应的刑期，使用整数>}}]}}, {{"name": "第二个被告", "judgment": [{{"standard_accusation": "第二个被告的第一个罪名", "imprisonment": "<对应的刑期，使用整数>"}}]}}]}}
"""
        return prompt


if __name__ == "__main__":
    import argparse
    import os

    from dotenv import load_dotenv

    from ..llm import get_llm
    from ..predictor import ZeroShotPredictor
    from ..utils import save_json
    from ..utils.data_utils import LegalCaseDataset

    load_dotenv()

    def get_data(split: str = "train") -> LegalCaseDataset:
        """Load the legal case dataset."""
        if split == "train":
            return LegalCaseDataset("./data/train.jsonl")
        elif split == "test":
            return LegalCaseDataset("./data/test.jsonl")
        else:
            raise ValueError("Invalid split. Use 'train' or 'test'.")

    parser = argparse.ArgumentParser(description="Zero-shot legal case prediction")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save the output results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (train or test)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the LLM model to use",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing cases from",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training cases to use for zero-shot prediction",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    split = args.split
    model_name = args.model_name
    start_index = args.start_index
    train_size = args.train_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    legal_data = get_data(split)
    if train_size is not None:
        legal_data = legal_data[start_index : start_index + train_size]
    else:
        legal_data = legal_data[start_index:]
    print(f"Loaded {len(legal_data)} cases from the dataset.")
    llm = get_llm(model_name=model_name)
    zero_shot_predictor = ZeroShotPredictor(llm)
    for i, data in enumerate(legal_data):
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        prompt = zero_shot_predictor.build_zero_shot_prompt(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["system_prompt"] = zero_shot_predictor.system_prompt
        result_to_save["prompt"] = prompt
        response, result = zero_shot_predictor.predict_judgment(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["response"] = response
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"{i + start_index}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")
