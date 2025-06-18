import argparse

from src.utils import save_jsonl
from src.utils.data_utils import ArticleLoader


def main():
    parser = argparse.ArgumentParser(description="Zero-shot legal case prediction")
    parser.add_argument(
        "--articles-file", type=str, required=True, help="Path of the articles file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="File to save the zero-shot prediction requests in JSONL format",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the LLM model to use"
    )
    args = parser.parse_args()
    model_name = args.model_name

    article_data = list(ArticleLoader(args.articles_file).all_articles.values())
    jsonl_data = []
    url_dict = {
        "text-embedding-v3": "/v1/embeddings",
    }
    for i, data in enumerate(article_data):
        result_to_save = {}
        result_to_save["custom_id"] = str(i + 1)
        result_to_save["method"] = "POST"
        result_to_save["url"] = url_dict[model_name]
        result_to_save["body"] = {
            "model": model_name,
            "input": data,
            "encoding_format": "float",
        }
        jsonl_data.append(result_to_save)
        print(f"Processed case {i + 1}/{len(article_data)}")
    save_jsonl(jsonl_data, args.output_file)
    print(f"Zero-shot prediction requests saved to {args.output_file}")


if __name__ == "__main__":
    main()
