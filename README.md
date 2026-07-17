# TTR Binding Activity Predictor

## Environment setup

Build the Docker image from the repository root.

```bash
docker build -t ttr_predictor_env ./create_docker_image
```

Start the Docker container.

```bash
docker run -it --gpus all \
  -v ${PWD}:/workspace \
  -w /workspace/predictor \
  ttr_predictor_env
```

For Windows PowerShell:

```powershell
docker run -it --gpus all `
  -v ${PWD}:/workspace `
  -w /workspace/predictor `
  ttr_predictor_env
```

## Input CSV

The input CSV must contain a `SMILES` column.

Example:

```csv
Chemical Name,SMILES
ethanol,CCO
benzene,c1ccccc1
```

## Remove invalid SMILES

```bash
python3.8 clean.py \
  --i input.csv \
  --o output.csv \
  --invalid compound_invalid_smiles.csv
```

## Run predictor

```bash
python3.8 predictor.py --i cleaned.csv
```

## Output

Prediction results are saved in a timestamped directory under `result/`.

Main output files:

```text
predictions.csv
predicted_y_histogram.png
mordred_descriptors.csv
```
