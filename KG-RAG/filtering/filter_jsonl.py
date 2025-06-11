from pathlib import Path
import argparse

from utils import read_jsonl_file, read_json_file, write_jsonl_file 


def filter(jsonl_path, filtering_file, output_path, name_postfix):

    filtering_dict = read_json_file(filtering_file)

    for key in filtering_dict.keys():
        
        file_name = key + name_postfix + ".jsonl"
        file_path = jsonl_path / file_name

        if not file_path.exists():
            raise ValueError(f"Path {file_path} does not exist")

        print(f"Processing {file_path}")

        dataset = read_jsonl_file(file_path)

        filtered_dataset = []
        for item in dataset:
            ix = item["rel_ix"].split("_")[1]
            if int(ix) in filtering_dict[key]:
                filtered_dataset.append(item)


        # Write the filtered dataset to a new JSONL file
        write_jsonl_file(filtered_dataset, output_path / file_name)
    

def main():
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--jsonl_dir", type=str, help="", default="")
    parser.add_argument("--filtering_file", type=str, help="", default="filteringfile.json")
    parser.add_argument("--output_dir", type=str, help="", default="")
    parser.add_argument("--name_postfix", type=str, help="", default="")

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_dir)

    if not jsonl_path.exists():
        raise ValueError(f"Path {args.jsonl_dir} does not exist")

    output_path = Path(args.output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    filter(jsonl_path, args.filtering_file, output_path, args.name_postfix)

    print("Done")

if __name__ == "__main__":
    main()
