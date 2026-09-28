
## Linters

```bash
mypy --strict --python-executable .venv/bin/python src/
```

```bash
flake8 --max-line-length=148 src/
```


## Testing

```bash
uv run -m scripts.get_test_predictions \
    --run_name RUN_NAME \
    --tokenizer-path "./checkpoints/cpt_6" \
    --state-dict-path "./checkpoints/cpt_6/state_dict.pt" \
    --backbone-model-id "cointegrated/rubert-tiny2" \
    --batch-size 128 \
    --device "cuda"
```