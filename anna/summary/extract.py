"""Extracts the performance metrics from TF events files"""

import os
import tensorflow as tf
from collections import defaultdict


def get_metrics(events_paths):
    """
    Parses the given `events_path` file, extracting all tagged metrics.

    Args:
        events_paths (list[str]): the paths to an tfevents files

    Returns:
        result (dict): keys are metric names, values are lists of (step, value)
    """
    result = defaultdict(list)
    for p in events_paths:
        for e in tf.train.summary_iterator(p):
            for v in e.summary.value:
                result[v.tag].append((e.step, v.simple_value))

    return result


def parse_all(models_dir):
    """
    Retrieves all metrics from all models in `models-dir`.

    Args:
        models_dir (str): path where all model runs are stored

    Returns:
        result (dict): keys are model names (taken from the folder name), values
          are dictionaries with 3 entries, train/val/test.
    """
    models = [(e.name, e.path) for e in os.scandir(models_dir) if e.is_dir()]

    result = {}
    for name, path in models:
        metrics = parse_model(path)
        if metrics:
            result[name] = metrics

    return result


def parse_model(model_dir):
    """
    Retrieves all metrics from the given model in `model_dir`.

    Args:
        model_dir (str): path where a model run is stored

    Returns:
        result (dict): keys are train/val/test, values are dictionaries of
          metrics and their values (output of `get_metrics()`).
    """
    train_events = find_events(model_dir)
    val_events = find_events(os.path.join(model_dir, "eval_val"))
    test_events = find_events(os.path.join(model_dir, "eval_test"))

    return {
        "train": get_metrics(train_events),
        "val": get_metrics(val_events),
        "test": get_metrics(test_events)
    }


def find_events(path):
    """
    Finds the tfevents file in the given `path`. Returns `ValueError` if no
    events file is found, or more than 1 exists.

    Args:
        path (str): a path that contains a model run

    Returns:
        events_paths (list[str]): paths to the tfevents file

    Raises:
        ValueError if no events file is found, or more than 1 exists
    """
    try:
        events = [e.path for e in os.scandir(path) if "events.out" in e.name]
    except FileNotFoundError:
        return []

    def events_sort_key(path):
        # Get the number in the path
        start = path.find("events.out.tfevents")
        start += len("events.out.tfevents.")
        end = path.find(".", start)

        return int(path[start:end])

    return sorted(events, key=events_sort_key)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: extract.py MODELS_FOLDER PATH_TO_SAVE")
        exit(1)

    # Resolve models folders and final metrics path
    models_path = os.path.abspath(sys.argv[1])
    save_path = os.path.abspath(sys.argv[2])

    # Get all metrics
    all_metrics = parse_all(models_path)

    # Save as json
    with open(save_path, "w") as f:
        json.dump(all_metrics, f)
