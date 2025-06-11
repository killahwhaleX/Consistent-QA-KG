from pathlib import Path
import json
import argparse
import random
import math

from utils import read_jsonl_file 

def generate(proportion_to_keep, path, output_file_name):
    seed = 9999
    random.seed(seed)

    filtering_dict = {}

    for file in list(path.glob("*.jsonl")):
        print(f"Processing {file}")

        # Read the JSONL file
        dataset = read_jsonl_file(file)

        indexes = random.sample(range(len(dataset)), math.ceil(len(dataset) * proportion_to_keep))

        filtering_dict[file.stem] = sorted(indexes)

    # Write the filtered dataset to a new JSONL file
    with open(output_file_name, "w") as f:
        f.write(json.dumps(filtering_dict) )
    

def main():
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--proportion_to_keep", type=float, help="", default=0.2)
    parser.add_argument("--trex_task_dir", type=str, help="trex data directory", default="")
    parser.add_argument("--output_file_name", type=str, help="", default="filteringfile.json")

    args = parser.parse_args()

    path = Path(args.trex_task_dir)

    if not path.exists():
        raise ValueError(f"Path {args.trex_task_dir} does not exist")

    generate(args.proportion_to_keep, path, args.output_file_name)

    print("Done")

if __name__ == "__main__":
    main()
