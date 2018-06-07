# Parse TF summaries

This module provised simple utilities to extract the important info from the
summary files created while training/evaluation.

## How to run

To extract all metrics, run:

```bash
python extract.py PATH_TO_MODELS out.json
```

To convert them to a latex table, run:

```bash
python utils.py out.json
```
