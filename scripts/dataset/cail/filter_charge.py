import argparse

from src.utils import load_json, save_json

parser = argparse.ArgumentParser("Filter the cail data charge to course")

parser.add_argument(
    "--cail-accu-file",
    type=str,
    required=True,
    help="The path of accu.txt of cail dataset",
)
parser.add_argument(
    "--course-charge-file",
    type=str,
    required=True,
    help="The pato of the charges.json of course dataset",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="The path to save the filtered charges.json of cail dataset",
)

args = parser.parse_args()

course_charge_dict = load_json(args.course_charge_file)

cail_charge_list = []

with open(args.cail_accu_file, encoding="utf-8") as f:
    for line in f.readlines():
        cail_charge_list.append(line.strip() + "罪")

common_charges = list(set(course_charge_dict.keys()) & set(cail_charge_list))

common_charges_dict = {}

current_id = 0

for charge in common_charges:
    common_charges_dict[charge[:-1]] = current_id
    current_id += 1

save_json(common_charges_dict, args.output)
