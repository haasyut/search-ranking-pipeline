import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from joblib import dump, load
from datetime import datetime

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_format_read import read_dataset
from utils.ndcg import validate

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
data_path = os.path.join(base_path, "data", "train")
test_path = os.path.join(base_path, "data", "test")

raw_data_path = os.path.join(data_path, "raw_train.txt")
x_ranksvm_path = os.path.join(data_path, "X_ranksvm.npy")
y_ranksvm_path = os.path.join(data_path, "y_ranksvm.npy")
model_path = os.path.join(base_path, "data", "model", "ranksvm_model.joblib")
test_data_path = os.path.join(test_path, "test.txt")

# Top-20 features based on LightGBM importance (0-based indices)
top_features = [204, 17, 35, 10, 147, 172, 44, 53, 26, 165, 244, 220, 219, 312, 116, 218, 213, 167, 250, 146]

def select_top_features_from_raw(tokens, top_features):
    all_features = [0.0] * (max(top_features) + 1)
    for item in tokens:
        if ':' not in item:
            continue
        idx, val = item.split(':')
        all_features[int(idx) - 1] = float(val)
    return [all_features[i] for i in top_features]

def process_data_for_ranksvm(input_path, x_output_path, y_output_path):
    print(f"Processing raw data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    query_groups = {}
    for line in lines:
        if '#' in line:
            line = line[:line.index('#')].strip()
        tokens = line.split()
        label = int(tokens[0])
        qid = tokens[1].split(':')[1]

        selected_features = select_top_features_from_raw(tokens[2:], top_features)

        if qid not in query_groups:
            query_groups[qid] = []
        query_groups[qid].append((label, np.array(selected_features)))

    X, y = [], []
    for qid, group in query_groups.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                label_i, feat_i = group[i]
                label_j, feat_j = group[j]
                if label_i == label_j:
                    continue
                diff = feat_i - feat_j
                X.append(diff)
                y.append(1 if label_i > label_j else -1)

                X.append(-diff)
                y.append(-1 if label_i > label_j else 1)

    X = np.array(X)
    y = np.array(y)
    print(f"Saving RankSVM training data: X shape = {X.shape}, y shape = {y.shape}")
    np.save(x_output_path, X)
    np.save(y_output_path, y)

def train_ranksvm(x_path, y_path, model_out_path):
    print("Training RankSVM model...")
    X = np.load(x_path)
    y = np.load(y_path)
    X, y = shuffle(X, y, random_state=42)
    clf = LinearSVC(dual=False, max_iter=10000)
    clf.fit(X, y)
    dump(clf, model_out_path)
    print(f"Model saved to {model_out_path}")

def predict_scores(test_X, model_path):
    clf = load(model_path)
    scores = clf.decision_function(test_X)
    return scores

def test_ndcg(test_file, model_path):
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    X, y, qids, comments = read_dataset(lines)
    X = np.array([
        select_top_features_from_raw(tokens[2:], top_features)
        for tokens in [line.strip().split() for line in lines if line.strip()]
    ])
    scores = predict_scores(X, model_path)
    avg_ndcg, _ = validate(qids, y, scores, 60)
    print("NDCG@60:", avg_ndcg)

def predict_and_print(test_file, model_path):
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    X, y, qids, comments = read_dataset(lines)
    X = np.array([
        select_top_features_from_raw(tokens[2:], top_features)
        for tokens in [line.strip().split() for line in lines if line.strip()]
    ])
    scores = predict_scores(X, model_path)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_comments = comments[sorted_indices]
    print("Top ranked documents:")
    for line in sorted_comments[:20]:
        print(line.strip())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ranksvm.py [-process | -train | -test | -predict | -ndcg]")
        sys.exit(0)

    if sys.argv[1] == '-process':
        process_data_for_ranksvm(raw_data_path, x_ranksvm_path, y_ranksvm_path)

    elif sys.argv[1] == '-train':
        train_ranksvm(x_ranksvm_path, y_ranksvm_path, model_path)

    elif sys.argv[1] == '-test':
        X = np.load(x_ranksvm_path)
        y = np.load(y_ranksvm_path)
        clf = load(model_path)
        acc = clf.score(X, y)
        print(f"Accuracy on pairwise data: {acc:.4f}")

    elif sys.argv[1] == '-predict':
        predict_and_print(test_data_path, model_path)

    elif sys.argv[1] == '-ndcg':
        test_ndcg(test_data_path, model_path)
