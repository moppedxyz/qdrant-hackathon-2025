import pandas as pd


def load_train_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_parquet("data/train_data.parquet")
        train_eval_data = pd.read_parquet("data/train_eval_data.parquet")
    except Exception as e:
        print(f"Error loading train data: {e}")
        raise Exception("Please run data/prepare.py to generate the necessary files.")
    return train_data, train_eval_data


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        domain_test_data = pd.read_parquet("data/domain_test_data.parquet")
        ood_test_data = pd.read_parquet("data/ood_test_data.parquet")
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise Exception("Please run data/prepare.py to generate the necessary files.")
    return domain_test_data, ood_test_data
