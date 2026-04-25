import argparse
import json
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "MATH" / "train.jsonl"
DEFAULT_VALIDATION_FILE = PROJECT_ROOT / "data" / "MATH" / "validation.jsonl"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MATH jsonl files into verl parquet format."
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-train-samples", type=int, default=None)
    parser.add_argument("--num-validation-samples", type=int, default=None)
    return parser.parse_args()

def load_prompt_template(prompt_file: Path) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
    
def convert_example(
        example: dict,
        idx: int,
        split: str,
        prompt_template: str,
) -> dict:
    question = example["problem"]
    ground_truth = example["answer"]

    data = {
        "data_source": "Math",
        "prompt":[
            {
                "role": "system",
                "content": prompt_template.split("User:")[0].strip()
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
         "extra_info": {
            "split": split,
            "index": idx,
            "subject": example["subject"],
            "level": example["level"],
            "unique_id": example["unique_id"],
         }
    }

    return data

def write_parquet(records, output_file):
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_file)

def load_split(
        input_file: Path,
        split,
        prompt_template,
        max_samples=None,
):
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            example = json.loads(line)
            record = convert_example(example, idx, split, prompt_template)
            records.append(record)

            if max_samples is not None and len(records) >= max_samples:
                break
    
    return records

    
def main():
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt_file)
    
    train_records = load_split(args.train_file, "train", prompt_template, args.num_train_samples)
    val_records = load_split(args.validation_file, "validation", prompt_template, args.num_validation_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_output_file = args.output_dir / "train.parquet"
    val_output_file = args.output_dir / "validation.parquet"

    write_parquet(train_records, train_output_file)
    write_parquet(val_records, val_output_file)

    print(f"Wrote {len(train_records)} records to {train_output_file}")
    print(f"Wrote {len(val_records)} records to {val_output_file}")

        

if __name__ == "__main__":
    main()
