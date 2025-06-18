import argparse

from src.utils import load_jsonl, save_jsonl

parser = argparse.ArgumentParser("Match articles embedding and text")
parser.add_argument(
    "--batch-file", type=str, required=True, help="The path of the batch file generated"
)
parser.add_argument(
    "--result-file",
    type=str,
    required=True,
    help="The path of the result file of the embedding",
)
parser.add_argument(
    "--output-file", type=str, required=True, help="The path of the output file"
)

args = parser.parse_args()

law_data = []

articles = load_jsonl(args.batch_file)
results = load_jsonl(args.result_file)

for article_record in articles:
    article_id = article_record["custom_id"]
    article_content = article_record["body"]["input"]
    for result_record in results:
        result_id = result_record["custom_id"]
        if article_id == result_id:
            embedding = result_record["response"]["body"]["data"][0]["embedding"]
            law_data.append(
                {"id": article_id, "article": article_content, "embedding": embedding}
            )

save_jsonl(law_data, args.output_file)
