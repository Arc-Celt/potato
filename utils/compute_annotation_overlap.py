"""
Compute overlap metrics between human annotations and model keywords.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def extract_model_keywords(instance: Dict) -> Set[str]:
    """Extract model keywords from original data instance."""
    return set(instance.get('model_keywords', []))


def extract_annotated_keywords(instance: Dict) -> Set[str]:
    """Extract annotated keywords from annotation instance."""
    label_annotations = instance.get('label_annotations', {})
    valid_keywords = label_annotations.get('valid_keywords', {})

    keywords = set()
    for keyword in valid_keywords.keys():
        if keyword != "None of the above":
            keywords.add(keyword)

    return keywords


def compute_metrics(predicted: Set[str], actual: Set[str]) -> Dict[str, float]:
    """Compute precision, recall, F1, and Jaccard similarity."""
    if not predicted and not actual:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'jaccard': 1.0
        }

    intersection = predicted & actual
    union = predicted | actual

    precision = len(intersection) / len(predicted) if predicted else 0.0
    recall = len(intersection) / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }


def compute_aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute aggregate statistics across all instances."""
    if not all_metrics:
        return {}

    aggregate = defaultdict(list)
    for metrics in all_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregate[key].append(value)

    result = {}
    for key, values in aggregate.items():
        if values:
            result[f'{key}_mean'] = sum(values) / len(values)
            result[f'{key}_std'] = (
                sum((x - result[f'{key}_mean'])**2 for x in values) / len(values)
            ) ** 0.5

    return result


def compute_user_metrics(user_annotations: List[Dict], original_lookup: Dict[str, Dict]) -> Dict:
    """Compute metrics for a single user."""
    user_metrics = []
    per_instance_results = []
    total_instances = 0

    for annotated_instance in user_annotations:
        instance_id = annotated_instance.get('instance_id') or annotated_instance.get('id')
        if not instance_id:
            continue

        if instance_id not in original_lookup:
            continue

        original_instance = original_lookup[instance_id]

        model_keywords = extract_model_keywords(original_instance)
        annotated_keywords = extract_annotated_keywords(annotated_instance)

        metrics = compute_metrics(annotated_keywords, model_keywords)

        total_instances += 1
        user_metrics.append(metrics)

        per_instance_results.append({
            'instance_id': instance_id,
            'model_keywords': sorted(list(model_keywords)),
            'annotated_keywords': sorted(list(annotated_keywords)),
            'metrics': metrics
        })

    aggregate_metrics = compute_aggregate_metrics(user_metrics)

    return {
        'total_instances': total_instances,
        'aggregate_metrics': aggregate_metrics,
        'per_instance': per_instance_results
    }


def main():
    original_data_path = "data/keyword_annotation_qwen3_32b_fp8.jsonl"
    annotated_data_path = "annotation_output/keyword_annotation_qwen3_32b_fp8/annotated_instances.jsonl"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "annotation_overlap_metrics.json")

    # Load data
    original_data = load_jsonl(original_data_path)
    original_lookup = {item['id']: item for item in original_data}

    annotated_data = load_jsonl(annotated_data_path)

    # Group annotations by user_id
    user_to_annotations = defaultdict(list)
    for annotated_instance in annotated_data:
        user_id = annotated_instance.get('user_id')
        if user_id:
            user_to_annotations[user_id].append(annotated_instance)

    # Compute metrics per user
    user_results = {}
    for user_id in sorted(user_to_annotations.keys()):
        user_annotations = user_to_annotations[user_id]
        user_result = compute_user_metrics(user_annotations, original_lookup)
        user_results[user_id] = user_result

        agg = user_result['aggregate_metrics']
        print(f"{user_id}: Precision={agg.get('precision_mean', 0):.3f}, "
              f"Recall={agg.get('recall_mean', 0):.3f}, "
              f"F1={agg.get('f1_mean', 0):.3f}, "
              f"Jaccard={agg.get('jaccard_mean', 0):.3f}")

    # Compute overall aggregate across all users
    all_instance_metrics = []
    for user_result in user_results.values():
        for inst_result in user_result['per_instance']:
            if 'metrics' in inst_result:
                all_instance_metrics.append(inst_result['metrics'])
    overall_metrics = compute_aggregate_metrics(all_instance_metrics)

    total_all_instances = sum(r['total_instances'] for r in user_results.values())

    print(f"\nOverall: Precision={overall_metrics.get('precision_mean', 0):.3f}, "
          f"Recall={overall_metrics.get('recall_mean', 0):.3f}, "
          f"F1={overall_metrics.get('f1_mean', 0):.3f}, "
          f"Jaccard={overall_metrics.get('jaccard_mean', 0):.3f}")

    # Save results
    output_data = {
        'overall': {
            'total_instances': total_all_instances,
            'aggregate_metrics': overall_metrics
        },
        'per_user': user_results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
