import argparse

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from joblib import dump

from model_classes import LGBMModel


def make_parser() -> argparse.ArgumentParser:
    '''
    Make cli arguments parser

    Returns:
        CLI args parser
    '''
    parser = argparse.ArgumentParser(
        description='Train catboost model'
    )

    parser.add_argument('-data', type=str,
                        default='train.csv',
                        help='preprocessed dataset filename')

    parser.add_argument('-version', type=int,
                        default=0,
                        help='model version')

    parser.add_argument('-lr', type=float,
                        default=0.1,
                        help='learning rate')

    parser.add_argument('-iters', type=int,
                        default=500,
                        help='number of iterations')

    parser.add_argument('-test_size', type=float,
                        default=0.2,
                        help='ratio of test part')

    parser.add_argument('-refit', type=bool,
                        default=True,
                        help='refit model on full dataset')

    return parser


def print_metrics(metrics: dict) -> None:
    '''
    Print metrics as a table

    Args:
        metrics: Dictionary with metrics
    '''
    print('{:<20} || {:<10}'.format('Metric', 'Value'))

    for metric in metrics.keys():
        print(f'{metric:<20} || {metrics[metric]:<10}')


def main() -> None:
    '''
    Main function responsible for catboost model training
    '''

    # Arguments parsing
    parser = make_parser()
    args = parser.parse_args()

    data_path = args.data
    version = args.version
    learning_rate = args.lr
    iterations = args.iters
    test_size = args.test_size
    refit = args.refit

    # Load preprocessed data
    data = pd.read_csv(
        f'../../data/processed/{data_path}',
        index_col='pair_id'
    )

    # Undersampling
    matched_count = data.is_duplicate.value_counts()[1]
    equal_df = pd.concat([data[data.is_duplicate == 1],data[data.is_duplicate == 0].sample(matched_count)])

    X = equal_df.drop(['is_duplicate'], axis=1)
    y = equal_df.is_duplicate

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size
    )

    # Model training on train dataset
    model = LGBMClassifier(
        num_threads=-1,
        boosting='gbdt',
        eta=learning_rate,
        n_iter=iterations
    )

    print('Fitting model on train dataset...')

    model.fit(X_train, y_train)

    # Train and test evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics_dict = {
        'precision (train)': metrics.precision_score(y_train, train_preds),
        'recall (train)': metrics.recall_score(y_train, train_preds),
        'f1 (train)': metrics.f1_score(y_train, train_preds, average='macro')
    }

    test_metrics_dict = {
        'precision (test)': metrics.precision_score(y_test, test_preds),
        'recall (test)': metrics.recall_score(y_test, test_preds),
        'f1 (test)': metrics.f1_score(y_test, test_preds, average='macro')
    }

    metrics_dict = train_metrics_dict | test_metrics_dict

    # If refit is True we train model on full dataset and evaluate it
    if refit:
        model = LGBMClassifier(
            num_threads=-1,
            boosting='gbdt',
            eta=learning_rate,
            n_iter=iterations
        )

        print('Refitting model on full dataset...')

        model.fit(X, y)

        full_preds = model.predict(X)

        full_metrics_dict = {
            'precision (full)': metrics.precision_score(y, full_preds),
            'recall (full)': metrics.recall_score(y, full_preds),
            'f1 (full)': metrics.f1_score(y, full_preds, average='macro')
        }

        metrics_dict |= full_metrics_dict

    print_metrics(metrics_dict)

    unificated_model = LGBMModel(model)

    # Save models
    dump(unificated_model, f'../../models/uni_lgbm_v{version}.joblib')
    dump(model, f'../../models/raw_lgbm_v{version}.joblib')


if __name__ == '__main__':
    main()
