import json
import random
import argparse
import pandas as pd
from pathlib import Path


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_bio_file(bio_file_path):
    """Load bio file in various formats."""
    if bio_file_path.endswith('.jsonl'):
        return load_jsonl(bio_file_path)
    elif bio_file_path.endswith('.json'):
        with open(bio_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif bio_file_path.endswith('.csv'):
        df = pd.read_csv(bio_file_path)
        return df.to_dict('records')
    else:
        raise ValueError("Unknown bio file format")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Potato keyword annotation"
    )
    parser.add_argument(
        "--model_outputs",
        default="model_outputs/char_personality_gpt5_mini_v2.jsonl",
        help="Path to model outputs JSONL"
    )
    parser.add_argument(
        "--bio_file",
        default=None,
        help=("Path to character biography file (JSON/JSONL/CSV). "
              "If not provided, uses char_bio_merged.jsonl in the same "
              "directory as model_outputs.")
    )
    parser.add_argument(
        "--output",
        default="data/keyword_annotation.jsonl",
        help="Output path for Potato data"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of characters to sample"
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=20,
        help="Total number of keywords to show per character"
    )

    args = parser.parse_args()

    print(f"Loading model outputs from {args.model_outputs}...")
    model_data = load_jsonl(args.model_outputs)
    model_lookup = {item['character_json']: item for item in model_data}

    # Use default bio file if not provided
    if args.bio_file is None:
        model_dir = Path(args.model_outputs).parent
        args.bio_file = str(model_dir / "char_bio_merged.jsonl")

    print(f"Loading bios from {args.bio_file}...")
    bio_data = load_bio_file(args.bio_file)

    joined_data = []
    all_keywords_pool = set()

    for bio_item in bio_data:
        char_id = (bio_item.get('character_json') or
                   bio_item.get('id'))
        if not char_id:
            continue

        if char_id in model_lookup:
            model_item = model_lookup[char_id]

            # Extract English keywords
            keywords = (model_item.get('personality_keywords', {})
                        .get('English', []))

            if keywords:
                name = (bio_item.get('character_name') or
                        bio_item.get('name', 'Unknown'))
                joined_data.append({
                    'id': char_id,
                    'name': name,
                    'biography': bio_item.get('biography', ''),
                    'model_keywords': keywords
                })
                all_keywords_pool.update(keywords)

    print(f"Matched {len(joined_data)} characters with models outputs.")

    # Sample
    if len(joined_data) > args.sample_size:
        print(f"Sampling {args.sample_size} characters.")
        sampled_data = random.sample(joined_data, args.sample_size)
    else:
        sampled_data = joined_data

    all_keywords_list = list(all_keywords_pool)
    output_rows = []

    for item in sampled_data:
        true_keywords = item['model_keywords']

        num_true = len(true_keywords)
        num_distractors = args.candidate_pool_size - num_true

        if num_distractors < 0:
            print(
                f"Warning: Character {item['id']} has {num_true} "
                f"keywords, truncating to {args.candidate_pool_size}"
            )
            candidates = random.sample(
                true_keywords, args.candidate_pool_size
            )
        else:
            possible_distractors = [
                k for k in all_keywords_list
                if k not in true_keywords
            ]
            if len(possible_distractors) < num_distractors:
                distractors = possible_distractors
            else:
                distractors = random.sample(
                    possible_distractors, num_distractors
                )

            candidates = true_keywords + distractors

        random.shuffle(candidates)

        display_text = (f"<h1>{item['name']}</h1>"
                        f"<p>{item['biography']}</p>")

        output_rows.append({
            "id": item['id'],
            "text": display_text,
            "candidates": candidates,
            "model_keywords": true_keywords,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"Successfully wrote {len(output_rows)} items to {args.output}")


if __name__ == "__main__":
    main()
