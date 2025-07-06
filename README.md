# search-ranking-pipeline

## Overview

This project is a learning-to-rank pipeline that simulates a real-world search ranking scenario, with each of the four main stages documented in detail in the [docs/](./docs) folder:

1. **[Label Generation](docs/01-label-generation.md)** – Convert user behavior logs into graded relevance labels.
2. **[Feature Engineering](docs/02-feature-engineering.md)** – Extract content-based features for query-document pairs.
3. **[Model Training](docs/03-model-training.md)** – Train ranking models (e.g., LightGBM, RankSVM).
4. **[Evaluation](docs/04-evaluation.md)** – Assess performance using ranking metrics and offline A/B testing.

The goal of this project is to build a modular, interpretable, and reproducible ranking system from end to end. It serves both as a practical implementation of learning-to-rank theory and a framework for benchmarking different ranking algorithms in offline evaluation settings.

## Getting Started
Follow the steps below to set up the environment and run the pipeline locally.
#### Prerequisites
```bash
Python >= 3.8
```

#### Clone the repository
```bash
git clone https://github.com/yourusername/search-ranking-pipeline.git
cd search-ranking-pipeline
```
#### Create a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Install dependencies
```bash
pip install -r requirements.txt 
```

#### Run the end-to-end pipeline (Optional)
You can either run each stage via CLI scripts or launch the full pipeline using the notebook:
```bash
pip install notebook ipykernel
jupyter notebook demo.ipynb
```

##  Command Line Interface (CLI)

Once installed, you can run the main pipeline script with different flags to execute specific steps:

```bash
python src/models/<model_type>.py <flag>
```

#### Classic LightGBM Workflow (lightgbm.py)

| Flag           | Description                                                              |
|----------------|---------------------------------------------------------------------------|
| `-process`     | Convert raw training data into `feats.txt` and `group.txt` (LightGBM format) |
| `-train`       | Train the LightGBM ranker and save the model                             |
| `-plottree`    | Visualize a specific decision tree in the trained model                  |
| `-predict`     | Predict scores on test set and print top-ranked documents                |
| `-ndcg`        | Evaluate model on test set using NDCG                                    |
| `-feature`     | Print traditional LightGBM feature importance                            |
| `-shap`        | Use SHAP to visualize and interpret feature importance                   |
| `-leaf`        | Output leaf indices for test samples and one-hot encode them             |

#### Neural Enhanced LightGBM + MLP (lightgbm.py)

| Flag           | Description                                                              |
|----------------|---------------------------------------------------------------------------|
| `-train_nn`    | Train LightGBM with an MLP (neural network) feature head                 |
| `-predict_nn`  | Predict top-ranked results using the hybrid model                        |
| `-ndcg_nn`     | Evaluate the hybrid model using NDCG on test data                        |

#### RankSVM Workflow (ranksvm.py)
| Flag           | Description                                                              |
|----------------|---------------------------------------------------------------------------|
| `-process`     | Preprocess raw ranking data into pairwise training format (`X`, `y`)     |
| `-train`       | Train RankSVM on pairwise data                                           |
| `-test`        | Evaluate RankSVM accuracy on training data                               |
| `-predict`     | Predict top-ranked results on test set                                   |
| `-ndcg`        | Evaluate RankSVM model using NDCG                                        |
| `-tune`        | Tune hyperparameters via grid search                                     |

### 🧪 Example Usage

```bash
# Step 1: Preprocess raw data
python src/models/lightgbm.py -process

# Step 2: Train LightGBM
python src/models/lightgbm.py -train

# Step 3: Evaluate performance
python src/models/lightgbm.py -ndcg

# Step 4: Train hybrid model (LightGBM + MLP)
python src/models/lightgbm.py -train_nn

# Step 5: Evaluate hybrid model
python src/models/lightgbm.py -ndcg_nn
```

> All model files and outputs will be saved to the `data/model/` and `data/plot/` directories by default.



## Code Structure
```
search-ranking-pipeline/
├── data/                      # Raw and processed data files
│   ├── model/                 # Model outputs
│   ├── plot/       
│   ├── test/                  
│   └── train/                
│
├── docs/                      # Step-by-step documentation for each stage
│   ├── 01-label-generation.md
│   ├── 02-feature-engineering.md
│   ├── 03-model-training.md
│   ├── 04-evaluation.md
│   └── images/             
│
├── src/                    
│   ├── features/              # Feature extraction logic
│   │   ├── base.py            
│   │   └── lgbmodule.py     
│   │
│   ├── models/                # Ranking models
│   │   ├── lightgbm.py        
│   │   ├── ranksvm.py        
│   │   └── ranksvm-feature.py 
│   │
│   └── utils/                 # Data I/O, parsing, and evaluation utils
│       ├── data_format_read.py
│       ├── data.py            
│       └── ndcg.py     
│
├── demo.ipynb                 # Run-through notebook demonstrating the full pipeline
├── requirement.txt       
└── README.md            
```

## Results and evaluation
*Visualizations to be add soon*

## Future work
- **Online evaluation or A/B testing simulation**  
  Simulate user feedback loop or deploy lightweight experiments for live ranking evaluation.

## Acknowledgments/References

- **LightGBM Ranker**: Ke et al. (2017), [paper](https://arxiv.org/abs/1711.08434)
- **RankSVM**: Joachims, T. (2002), [paper](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf)
- **LETOR Dataset**: Microsoft Research, [official site](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)
- Thanks to the open-source community for foundational implementations and tutorials.


## License

This project is licensed under the MIT License