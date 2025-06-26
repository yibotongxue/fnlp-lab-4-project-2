import argparse
import random

from src.utils import load_jsonl, save_jsonl

parser = argparse.ArgumentParser("Split the jsonl data to train and eval set")

parser.add_argument(
    "--input-file", type=str, required=True, help="The path of the jsonl data to split"
)
parser.add_argument(
    "--train-file", type=str, required=True, help="The path of the train set to save"
)
parser.add_argument(
    "--eval-file", type=str, required=True, help="The path of the eval set to save"
)
parser.add_argument(
    "--eval-size", type=float, default=0.05, help="The size of the eval set"
)

args = parser.parse_args()

all_data = load_jsonl(args.input_file)

random.shuffle(all_data)

if int(args.eval_size) == args.eval_size:
    assert (
        args.eval_size > 0
    ), f"The eval size should be larger than 1 because it represent the size of the eval set, now it is {int(args.eval_size)}"
    eval_size = int(args.eval_size)
else:
    assert (
        args.eval_size < 1.0
    ), f"The eval size should be less than 1.0 because it represent the percentage of the all data, now the eval size is {args.eval_size}"
    eval_size = int(len(all_data) * args.eval_size)

save_jsonl(all_data[:-eval_size], args.train_file)
save_jsonl(all_data[-eval_size:], args.eval_file)

print(f"Successfully split the data to train and eval set")
