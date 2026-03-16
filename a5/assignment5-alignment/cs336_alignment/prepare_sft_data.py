import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

DEFAULT_INPUT_PATH = project_root / "data" / "MATH" / "sft.jsonl"
DEFAULT_OUTPUT_PATH = project_root / "data" / "MATH" / "sft_filtered.jsonl"
DEFAULT_STATS_PATH = project_root / "data" / "MATH" / "sft_filtered_stats.json"


def filter_math_sft_dataset(
    input_file: str | Path,
    output_file: str | Path,
    stats_file: str | Path | None = None,
) -> dict[str, int]:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    kept_examples = 0
    format_failures = 0
    wrong_answers = 0

    with input_file.open("r", encoding="utf-8") as in_f, output_file.open(
        "w",
        encoding="utf-8",
    ) as out_f:
        for line in in_f:
            total_examples += 1
            example = json.loads(line)
            ground_truth = example.get("ground_truth")
            if ground_truth is None:
                raise ValueError(
                    "Expected each SFT example to include a `ground_truth` field for filtering."
                )

            scores = r1_zero_reward_fn(example["response"], ground_truth)
            if scores["answer_reward"] == 1.0:
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept_examples += 1
            elif scores["format_reward"] == 0.0:
                format_failures += 1
            else:
                wrong_answers += 1

    stats = {
        "input_examples": total_examples,
        "kept_examples": kept_examples,
        "filtered_examples": total_examples - kept_examples,
        "format_failures": format_failures,
        "wrong_answers": wrong_answers,
    }

    if stats_file is not None:
        stats_file = Path(stats_file)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the original MATH SFT dataset.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the filtered MATH SFT dataset.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Path to write filtering statistics as JSON.",
    )
    args = parser.parse_args()

    stats = filter_math_sft_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        stats_file=args.stats_file,
    )

    print(f"Input examples: {stats['input_examples']}")
    print(f"Kept examples: {stats['kept_examples']}")
    print(f"Filtered examples: {stats['filtered_examples']}")
    print(f"Format failures: {stats['format_failures']}")
    print(f"Wrong answers: {stats['wrong_answers']}")
    print(f"Filtered dataset written to: {args.output_file}")
