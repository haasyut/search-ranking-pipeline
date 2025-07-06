import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import lightgbm as lgb
from sklearn import datasets as ds
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import torch


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath("src"))

from sklearn.preprocessing import OneHotEncoder
from utils.data_format_read import read_dataset
from utils.ndcg import validate
import shap
import matplotlib.pyplot as plt
import graphviz

from features.lgbmodule import LightGBMWithNN


def split_data_from_keyword(data_read, data_group, data_feats):
    """
    Convert raw training data into the format required by LightGBM for ranking tasks.

    Parameters:
    - data_read: Path to the input raw data file (containing label, query id, and features)
    - data_group: Path to the output group file, indicating number of samples per query
    - data_feats: Path to the output feature file, each line represents one training sample
    """
    with open(data_group, 'w', encoding='utf-8') as group_path:
        with open(data_feats, 'w', encoding='utf-8') as feats_path:
            dataframe = pd.read_csv(data_read,
                                    sep=' ',
                                    header=None,
                                    encoding="utf-8",
                                    engine='python')
            current_keyword = ''
            current_data = []
            group_size = 0
            for _, row in dataframe.iterrows():
                feats_line = [str(row[0])]
                for i in range(2, len(dataframe.columns) - 1):
                    feats_line.append(str(row[i]))
                if current_keyword == '':
                    current_keyword = row[1]
                if row[1] == current_keyword:
                    current_data.append(feats_line)
                    group_size += 1
                else:
                    for line in current_data:
                        feats_path.write(' '.join(line))
                        feats_path.write('\n')
                    group_path.write(str(group_size) + '\n')

                    group_size = 1
                    current_data = []
                    current_keyword = row[1]
                    current_data.append(feats_line)

            for line in current_data:
                feats_path.write(' '.join(line))
                feats_path.write('\n')
            group_path.write(str(group_size) + '\n')

def save_data(group_data, output_feature, output_group):
    """
    Save one group of training samples to feature and group files.

    Parameters:
    - group_data: A list of samples belonging to the same query group.
                  Each sample is a list: [label, qid, feature1, feature2, ...]
    - output_feature: File handle to write features (feats.txt)
    - output_group: File handle to write group sizes (group.txt)
    """
    if len(group_data) == 0:
        return
    output_group.write(str(len(group_data)) + '\n')
    for data in group_data:
        # Filter out features with value 0.0
        # feats = [p for p in data[2:] if float(p.split(":")[1]) != 0.0]
        feats = [p for p in data[2:] if float(p.split(":")[1]) != 0.0]
        output_feature.write(data[0] + ' ' + ' '.join(feats) + '\n') # data[0] => level ; data[2:] => feats

def process_data_format(test_path, test_feats, test_group):
    """
    Convert raw ranking data into LightGBM-compatible input format.

    Parameters:
    - test_path: Path to raw data file (e.g., LETOR format with qid and features)
    - test_feats: Output file path for features (feats.txt)
    - test_group: Output file path for group sizes (group.txt)
    """
    with open(test_path, 'r', encoding='utf-8') as f_read:
        with open(test_feats, 'w', encoding='utf-8') as output_feature:
            with open(test_group, 'w', encoding='utf-8') as output_group:
                group_data = []
                group = ''
                for line in f_read:
                    if '#' in line:
                        line = line[:line.index('#')]
                    splits = line.strip().split()
                    if splits[1] != group: # qid => splits[1]
                        save_data(group_data, output_feature, output_group)
                        group_data = []
                        group = splits[1]
                    group_data.append(splits)
                save_data(group_data, output_feature, output_group)

def load_data(feats, group):
    """
    Load preprocessed LightGBM training data from feature and group files.

    Parameters:
    - feats: Path to the feature file (in SVMlight format)
    - group: Path to the group file (indicating query group sizes)

    Returns:
    - x_train: Feature matrix (sparse format)
    - y_train: Label vector
    - q_train: Query group sizes
    """
    x_train, y_train = ds.load_svmlight_file(feats)  # scikit-learn built-in function
    q_train = np.loadtxt(group)  # each line is an integer group size
    return x_train, y_train, q_train

def load_data_from_raw(raw_data):
    """
    Load raw ranking dataset (LETOR format or similar).

    Parameters:
    - raw_data: File path to raw test data (with qid, features, and comments)

    Returns:
    - test_X: Feature matrix (as ndarray or sparse)
    - test_y: Label vector
    - test_qids: Query IDs
    - comments: Raw line suffixes (like document IDs or metadata)
    """
    with open(raw_data, 'r', encoding='utf-8') as testfile:
        test_X, test_y, test_qids, comments = read_dataset(testfile)
    return test_X, test_y, test_qids, comments

def train(x_train, y_train, q_train, model_save_path):
    """
    Train and save a LightGBM LambdaRank model.

    Parameters:
    - x_train: Feature matrix (sparse or dense)
    - y_train: Labels (relevance scores)
    - q_train: Query group sizes
    - model_save_path: File path to save the trained model

    Returns:
    - None
    """
    # Prepare training data in LightGBM format with group info
    train_data = lgb.Dataset(x_train, label=y_train, group=q_train)

    # Define model parameters for pairwise learning to rank (LambdaRank)
    params = {
        'task': 'train',                      # Task type
        'boosting_type': 'gbrt',              # Base learner: Gradient Boosting Trees
        'objective': 'lambdarank',            # Learning-to-rank objective
        'metric': 'ndcg',                     # Evaluation metric
        'ndcg_at': [10],                      # Top-N ranking metric
        'max_position': 10,                   # Used in NDCG calculation (safe to ignore warning)
        'metric_freq': 1,                     # Frequency of metric output
        'train_metric': True,                 # Show metrics during training
        'max_bin': 255,                       # Max number of bins for feature discretization
        'num_iterations': 200,                # Number of boosting rounds
        'learning_rate': 0.01,                # Step size shrinkage
        'num_leaves': 31,                     # Max number of leaves in one tree
        'tree_learner': 'serial',             # Single-machine learner
        'min_data_in_leaf': 30,               # Minimum samples per leaf
        'verbose': 2                          # Verbosity level
    }

    # Train the model using all data as training set
    gbm = lgb.train(params, train_data, valid_sets=[train_data])

    # Save model to disk
    gbm.save_model(model_save_path)

def plot_tree(model_path, tree_index, save_plot_path):
    """
    Visualize a specific decision tree from a trained LightGBM model.

    Parameters:
    - model_path: Path to the saved LightGBM model (.txt)
    - tree_index: Index of the tree to be visualized
    - save_plot_path: Path to save the rendered graph (without file extension)

    Returns:
    - None
    """
    if not os.path.exists(model_path):
        print(f"File does not exist: {model_path}")
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)
    graph = lgb.create_tree_digraph(gbm, tree_index=tree_index, name=f'tree{tree_index}')
    graph.render(filename=save_plot_path, view=True)  # Save and open the visualization

def predict(x_test, comments, model_input_path):
    """
    Predict ranking scores and return comments sorted by score.

    Parameters:
    - x_test: Feature matrix for testing
    - comments: Original comment or metadata strings (e.g. doc IDs, titles)
    - model_input_path: Path to the trained LightGBM model (.txt)

    Returns:
    - Sorted comments based on descending predicted scores
    """
    gbm = lgb.Booster(model_file=model_input_path)  # Load trained model
    ypred = gbm.predict(x_test)                     # Predict scores
    predicted_sorted_indexes = np.argsort(ypred)[::-1]  # Indices sorted by score descending
    t_results = comments[predicted_sorted_indexes]      # Reorder comments accordingly
    return t_results

def test_data_ndcg(model_path, test_path):
    """
    Evaluate the ranking performance of a LightGBM model using NDCG.

    Parameters:
    - model_path: Path to the trained LightGBM model (.txt or .mod file)
    - test_path: Path to the raw test dataset (in LETOR-style format)

    Returns:
    - None (prints average NDCG score)
    """
    with open(test_path, 'r', encoding='utf-8') as testfile:
        test_X, test_y, test_qids, comments = read_dataset(testfile)

    gbm = lgb.Booster(model_file=model_path)
    test_predict = gbm.predict(test_X)

    average_ndcg, _ = validate(test_qids, test_y, test_predict, 60)  # top-60
    print("All QIDs average NDCG:", average_ndcg)
    return average_ndcg

def plot_print_feature_shap(model_path, data_feats, plot_type):
    """
    Visualize feature importance and interactions using SHAP for a trained LightGBM model.

    Parameters:
    - model_path: Path to the trained LightGBM model
    - data_feats: Path to the training feature file in SVMlight format
    - plot_type: Integer (1–4) indicating the SHAP plot type:
        1 = summary bar and dot plot
        2 = single feature dependence plot
        3 = feature interaction plot
        4 = interaction summary plot

    Returns:
    - None (shows SHAP plots)
    """
    if not (os.path.exists(model_path) and os.path.exists(data_feats)):
        print(f"File does not exist: {model_path}, {data_feats}")
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)
    gbm.params["objective"] = "regression"  # Needed by SHAP

    # Simulate 46 feature names (feat0name, feat1name, ..., feat45name)
    feats_col_name = [f'feat{i}name' for i in range(46)]

    # Load features in SVMlight format
    X_train, _ = ds.load_svmlight_file(data_feats)
    feature_mat = X_train.todense()
    df_feature = pd.DataFrame(feature_mat, columns=feats_col_name)

    explainer = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(df_feature)

    # 1: Overall SHAP summary (bar + dot)
    if plot_type == 1:
        shap.summary_plot(shap_values, df_feature, plot_type="bar")
        shap.summary_plot(shap_values, df_feature)

    # 2: Single feature SHAP dependence plot
    elif plot_type == 2:
        shap.dependence_plot('feat3name', shap_values, df_feature)

    # 3: SHAP interaction between two features
    elif plot_type == 3:
        shap.dependence_plot('feat3name', shap_values, df_feature, interaction_index='feat5name')

    # 4: Summary of pairwise feature interactions
    elif plot_type == 4:
        shap_interaction_values = explainer.shap_interaction_values(df_feature)
        shap.summary_plot(shap_interaction_values, df_feature, max_display=4)


def plot_print_feature_importance(model_path):
    """
    Print feature importance of a trained LightGBM model.

    Parameters:
    - model_path: Path to the trained LightGBM model (.txt or .mod)

    Notes:
    - Maps LightGBM internal feature names like 'Column_0' to human-readable feature names like 'feat0name'.
    - Importance is based on 'split' count (i.e., how many times a feature is used in a split).
    """

    # Map from 'Column_#' to custom feature names (e.g., 'feat0name', 'feat1name', ...)
    feats_dict = {
        f"Column_{i}": f"feat{i}name" for i in range(46)
    }

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        sys.exit(0)

    gbm = lgb.Booster(model_file=model_path)

    # Get feature importance scores (based on split count)
    importances = gbm.feature_importance(importance_type='split')
    feature_names = gbm.feature_name()

    total_importance = sum(importances)

    for feature_name, importance in zip(feature_names, importances):
        if importance > 0:
            # Extract feature index from 'Column_#'
            feat_id = int(feature_name.split('_')[1]) + 1
            print(f"{feat_id} : {feats_dict.get(feature_name, feature_name)} : {importance} : {importance / total_importance:.4f}")

def get_leaf_index(data, model_path):
    """
    Extract the leaf indices from a trained LightGBM model and convert them into one-hot encoded vectors.

    Parameters:
    - data: Input feature matrix (e.g., from test or train set), in the same format used for training.
    - model_path: Path to the trained LightGBM model.

    Returns:
    - x_one_hot: One-hot encoded leaf indices (sparse matrix).
    """

    # Load the trained model
    gbm = lgb.Booster(model_file=model_path)

    # Predict leaf indices: shape = [n_samples, n_trees]
    ypred = gbm.predict(data, pred_leaf=True)

    # One-hot encode leaf indices (each tree's leaf treated as a category)
    one_hot_encoder = OneHotEncoder()
    x_one_hot = one_hot_encoder.fit_transform(ypred)

    # Print shape and dense version (for debug)
    print(x_one_hot.shape)
    print(x_one_hot.toarray())

    return x_one_hot

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python lgb_ltr.py [-process | -train | -plottree | -predict | -ndcg | -feature | -shap | -leaf]")
        sys.exit(0)

    # Get project root directory from current script's location
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Define data paths
    train_path = os.path.join(base_path, "data", "train")
    raw_data_path = os.path.join(train_path, "raw_train.txt")
    data_feats = os.path.join(train_path, "feats.txt")
    data_group = os.path.join(train_path, "group.txt")

    # Define model and plot output paths
    model_path = os.path.join(base_path, "data", "model", "model.mod")
    save_plot_path = os.path.join(base_path, "data", "plot", "tree_plot")

    if sys.argv[1] == '-process':
        # Convert input training data (similar to RankLib format) into LightGBM-required format:
        # - Save features and group sizes into separate files.
        # Example Input:
        #   1 qid:0 1:0.2 2:0.4 ... #comment
        #   2 qid:0 1:0.1 2:0.2 ...
        #   1 qid:1 1:0.2 2:0.1 ...
        #   ...
        # Output:
        #   feats.txt:
        #       1 1:0.2 2:0.4 ...
        #       2 1:0.1 2:0.2 ...
        #       ...
        #   group.txt:
        #       2
        #       4
        process_data_format(raw_data_path, data_feats, data_group)

    elif sys.argv[1] == '-train':
        # Train and save the model
        train_start = datetime.now()
        x_train, y_train, q_train = load_data(data_feats, data_group)
        train(x_train, y_train, q_train, model_path)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-plottree':
        # Visualize a specific decision tree in the model
        plot_tree(model_path, 2, save_plot_path)

    elif sys.argv[1] == '-predict':
        # Predict and sort results on test set (similar format to RankLib)
        train_start = datetime.now()
        predict_data_path = os.path.join(base_path, "data", "test", "test.txt")
        test_X, test_y, test_qids, comments = load_data_from_raw(predict_data_path)
        t_results = predict(test_X, comments, model_path)
        print(t_results)
        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))

    elif sys.argv[1] == '-ndcg':
        # Evaluate the model on test set using NDCG
        test_path = os.path.join(base_path, "data", "test", "test.txt")
        test_data_ndcg(model_path, test_path)

    elif sys.argv[1] == '-feature':
        # Print traditional feature importance
        plot_print_feature_importance(model_path)

    elif sys.argv[1] == '-shap':
        # Use SHAP to interpret feature importance
        plot_print_feature_shap(model_path, data_feats, 3)

    elif sys.argv[1] == '-leaf':
        # Get leaf node indices for samples and one-hot encode them
        raw_data = os.path.join(base_path, "data", "test", "leaf.txt")
        with open(raw_data, 'r', encoding='utf-8') as testfile:
            test_X, test_y, test_qids, comments = read_dataset(testfile)
        get_leaf_index(test_X, model_path)

    elif sys.argv[1] == '-train_nn':
        from features.lgbmodule import LightGBMWithNN
        import joblib

        print("Loading training data...")
        train_start = datetime.now()

        x_train, y_train, q_train = load_data(data_feats, data_group)

        print("Training LightGBM + MLP pipeline...")
        model = LightGBMWithNN(input_dim=x_train.shape[1])
        model.fit(x_train, y_train, q_train)

        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        train_end = datetime.now()
        consume_time = (train_end - train_start).seconds
        print("consume time : {}".format(consume_time))
    
    elif sys.argv[1] == '-predict_nn':
        from features.lgbmodule import LightGBMWithNN
        import joblib

        predict_data_path = os.path.join(base_path, "data", "test", "test.txt")
        test_X, test_y, test_qids, comments = load_data_from_raw(predict_data_path)

        model = joblib.load(model_path)
        scores = model.predict(test_X)

        print("Top results:")
        sorted_indices = np.argsort(scores)[::-1]
        for i in sorted_indices[:20]:
            print(comments[i].strip())
    
    elif sys.argv[1] == '-ndcg_nn':
        from features.lgbmodule import LightGBMWithNN
        import joblib

        print("Evaluating LightGBM + MLP model on NDCG...")
        test_path = os.path.join(base_path, "data", "test", "test.txt")
        test_X, test_y, test_qids, comments = load_data_from_raw(test_path)

        model = joblib.load(model_path)
        scores = model.predict(test_X)

        avg_ndcg, _ = validate(test_qids, test_y, scores, 60)
        print("NDCG@60:", avg_ndcg)
