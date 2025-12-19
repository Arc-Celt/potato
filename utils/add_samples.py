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
        description="Add more samples to existing annotation file"
    )
    parser.add_argument(
        "--existing_file",
        required=True,
        help="Path to existing annotation JSONL file"
    )
    parser.add_argument(
        "--model_outputs",
        default="model_outputs/char_personality_qwen3_32b_fp8.jsonl",
        help="Path to model outputs JSONL"
    )
    parser.add_argument(
        "--bio_file",
        default=None,
        help="Path to character biography file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of new samples to add"
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=20,
        help="Total number of keywords to show per character"
    )

    args = parser.parse_args()

    # Load existing file to get used character IDs
    print(f"Loading existing annotations from {args.existing_file}...")
    existing_data = load_jsonl(args.existing_file)
    used_ids = {item['id'] for item in existing_data}
    print(f"Found {len(used_ids)} existing samples")

    # Load model outputs and bio data (same logic as prepare_data.py)
    print(f"Loading model outputs from {args.model_outputs}...")
    model_data = load_jsonl(args.model_outputs)
    model_lookup = {item['character_json']: item for item in model_data}

    # Use default bio file if not provided
    if args.bio_file is None:
        model_dir = Path(args.model_outputs).parent
        args.bio_file = str(model_dir / "char_bio_merged.jsonl")

    print(f"Loading bios from {args.bio_file}...")
    bio_data = load_bio_file(args.bio_file)

    # Join data and filter out already-used characters
    available_data = []
    all_keywords_pool = set()

    for bio_item in bio_data:
        char_id = (bio_item.get('character_json') or
                   bio_item.get('id'))
        if not char_id or char_id in used_ids:
            continue

        if char_id in model_lookup:
            model_item = model_lookup[char_id]

            # Extract English keywords
            keywords = (model_item.get('personality_keywords', {})
                        .get('English', []))

            if keywords:
                name = (bio_item.get('character_name') or
                        bio_item.get('name', 'Unknown'))
                available_data.append({
                    'id': char_id,
                    'name': name,
                    'biography': bio_item.get('biography', ''),
                    'model_keywords': keywords
                })
                all_keywords_pool.update(keywords)

    print(f"Found {len(available_data)} available characters (excluding "
          f"already-used ones)")

    if len(available_data) < args.num_samples:
        print(f"Warning: Only {len(available_data)} available characters, "
              f"but {args.num_samples} requested. Using all available.")
        sampled_data = available_data
    else:
        print(f"Sampling {args.num_samples} characters.")
        sampled_data = random.sample(available_data, args.num_samples)

    # Generate candidates for new samples
    all_keywords_list = list(all_keywords_pool)
    new_rows = []

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

        candidates.append("None of the above")

        display_text = (f"<h1>{item['name']}</h1>"
                        f"<p>{item['biography']}</p>")

        new_rows.append({
            "id": item['id'],
            "text": display_text,
            "candidates": candidates,
            "model_keywords": true_keywords,
        })

    # Append to existing file
    print(f"Appending {len(new_rows)} new samples to {args.existing_file}...")
    with open(args.existing_file, 'a', encoding='utf-8') as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f"Successfully added {len(new_rows)} new samples. "
          f"Total samples: {len(existing_data) + len(new_rows)}")


if __name__ == "__main__":
    main()
