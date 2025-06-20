from typing import override

from .llm import LLMPredictor
from ..llm import BaseLLM
from ..retriever import BaseRetriever


class RagPredictor(LLMPredictor):
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever, law_count: int = 3):
        super().__init__(llm)
        self.retriever = retriever
        self.law_count = law_count

    @override
    def build_prompt(self, fact: str, defendants: list[str]) -> str:
        """
        Build a rag prompt for the LLM based on the provided fact and defendants.

        Args:
            fact (str): The input fact to include in the prompt.
            defendants (list[str]): A list of defendants to include in the prompt.

        Returns:
            str: The constructed prompt string.
        """
        related_laws = self.retriever.retrieve(fact, self.law_count)
        prompt = f"""你是一个中华人民共和国的法官，你现在需要根据检察院对多名被告的指控和一些相关的刑法条款，依据中华人民共和国刑法，判断每个被告犯下的所有罪名，并给出每个罪名的量刑，量刑以月为单位。进行判决的时候，你应该注意：
1. 你要完整、仔细的阅读检察官的指控，你的量刑必须严格的按照检察官的指控事实作出，不能脱离事实，也不必怀疑检察官的指控
2. 你应该按照中华人民共和国刑法的法律条款判决，每个被告的罪名都必须在刑法中有所对应，比如“伪证罪”
3. 所有罪名都必须在指控中有明显的直接体现，不能通过主观臆断推定
4. 每个罪名的量刑必须在中华人民共和国刑法的规定范围内，必须严格的按照法律条款来量刑
5. 你需要给出每个罪名的量刑，量刑以月为单位，量刑需要是整数，注意你需要给出完整的判决，包括有期徒刑、罚金等等，但最终的总结中你只需要给出有期徒刑的量刑，其他的刑罚不必考虑，如果不必判处有期徒刑（这是很可能的），则量刑为0
6. 特别注意被告犯下罪行的时间，如果对应的法律条款后续有修改，要遵从从旧兼从轻的原则
7. 不同被告可能会有不同的罪名，一个罪行不一定所有被告都会涉及，同时即使同一个罪名不同的被告也可能会有不同的量刑，你需要逐一分析，不能一概而论
8. 你的所有罪名都必须是完整的罪名，不能使用缩写或简称
9. 我们会给出相关的{self.law_count}条法律条文，这是你判断罪名和量刑的重要依据，但注意可能只有第一条或前若干条是真正相关的，也有可能都是不相关的，排在越靠前的相关性越高。如果你确信法律条文与指控事实不相关，或者不是主要的罪名，你可以根据自己的相关知识作出正确的罪名和量刑判断
以下是检察官的指控：
{fact}
以下是相关的法律条文：
{related_laws}
以下是被告的名单：
{', '.join(defendants)}
请你根据以上信息，给出每个被告的罪名和量刑，在此之前你应该进行详细的分析，对每一个被告要分析指控中他涉及的部分，注意被告可能同时被指控多项罪行，然后分析他可能涉及的罪名，首先分析我们给出的法条是否是进行定罪和判刑的合理法条，如果不是的，请重新尝试根据刑法知识分析，并注意要引用中华人民共和国刑法的法律条文，然后按照《关于常见犯罪的量刑指导意见（试行）》对他的每一条成立的罪名根据罪行和法律条文分析量刑，最后使用JSON格式对之前的分析进行总结，一个示例如下：
{{"outcomes": [{{"name": "第一个被告姓名", "judgment": [{{"standard_accusation": "第一个被告第一个罪名", "imprisonment": <对应的刑期，使用整数>}}, {{"standard_accusation": "第一个被告的第二个罪名", "imprisonment": <对应的刑期，使用整数>}}]}}, {{"name": "第二个被告", "judgment": [{{"standard_accusation": "第二个被告的第一个罪名", "imprisonment": "<对应的刑期，使用整数>"}}]}}]}}
"""
        return {
            "prompt": prompt,
            "laws": related_laws,
        }


if __name__ == "__main__":
    import argparse
    import os

    from dotenv import load_dotenv
    from tqdm import tqdm

    from ..llm import get_llm
    from ..utils import save_json
    from ..embed import BaseEmbedding, get_embedding_model
    from ..retriever import BaseRetriever, get_retriever
    from ..utils.data_utils import ArticleLoader, LegalCaseDataset

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
        "--articles-file",
        type=str,
        default="./data/articles.json",
        help="Path of the articles file",
    )
    parser.add_argument(
        "--embed-file",
        type=str,
        default="./data/articles_embed.jsonl",
        help="Path of the articles embedding file",
    )
    parser.add_argument(
        "--retriever-type", type=str, default="numpy", help="Type of retriever"
    )
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
        "--embedding-model",
        type=str,
        default="text-embedding-v3",
        help="The name of embedding model",
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
    articles_file = args.articles_file
    embed_file = args.embed_file
    retriever_type = args.retriever_type
    output_dir = args.output_dir
    split = args.split
    model_name = args.model_name
    embedding_model_name = args.embedding_model
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
    article_loader = ArticleLoader(articles_file, embed_file)
    if retriever_type == "numpy":

        def get_text(ids: int | list[int] | list[list[int]]):
            if isinstance(ids, int):
                return article_loader.embed_data[ids]["article"]
            elif isinstance(ids, list):
                return [get_text(index) for index in ids]
            else:
                raise TypeError(
                    f"The ids should be either int or list, but is {type(ids)}"
                )

        embedding_model: BaseEmbedding = get_embedding_model(
            model_name=embedding_model_name
        )
        retriever: BaseRetriever = get_retriever(
            retriever_type,
            embedder=embedding_model,
            text_getter=get_text,
            embedding_matrix=article_loader.embedding_matrix,
        )
    else:
        raise ValueError("Unsupported retriever type")
    rag_predictor = RagPredictor(llm, retriever, law_count=3)
    for i in tqdm(range(len(legal_data)), "Legal Judgement Predition"):
        data = legal_data[i]
        result_to_save = {}
        result_to_save["input"] = data.model_dump()
        prompt = rag_predictor.build_prompt(fact=data.fact, defendants=data.defendants)
        result_to_save["system_prompt"] = LLMPredictor.system_prompt
        result_to_save["prompt"] = prompt["prompt"]
        result_to_save["laws"] = prompt["laws"]
        response, result = rag_predictor.predict_judgment(
            fact=data.fact, defendants=data.defendants
        )
        result_to_save["response"] = response
        result_to_save["result"] = [outcome.model_dump() for outcome in result]
        save_json(result_to_save, os.path.join(output_dir, f"{i + start_index}.json"))
        print(f"Processed case {i + 1}/{len(legal_data)}")
