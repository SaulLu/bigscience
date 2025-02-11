import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_ratios_path",
        type=str,
        required=True,
        help="path to JSON file containing input dataset ratios. Values ares dictionary: {'dataset_path': str, 'ratio': float}",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"]

    )
    return parser.parse_args()

TOKEN_RANGES={
    "train": "0:0.949",
    "valid": "0.949:0.999",
    "test": "0.999:1.0"
}

def main():
    args = get_args()

    token_range = TOKEN_RANGES[args.split]

    with open(args.dataset_ratios_path, "r") as fi:
        ds_ratios = json.load(fi)

    list_string = []
    for ds_ratio in ds_ratios:
        elt_string = f"{ds_ratio['ratio']} {token_range} {ds_ratio['dataset_path']}"
        list_string.append(elt_string)

    # TODO: you can add some extra dataset names for validation/test
    print(f"\"{args.split}: " + ", ".join(list_string) + "\"")


if __name__ == "__main__":
    main()
