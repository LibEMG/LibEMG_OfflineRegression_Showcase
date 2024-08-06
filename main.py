from argparse import ArgumentParser

import matplotlib.pyplot as plt
from libemg.offline_metrics import OfflineMetrics
from libemg.datasets import OneSubjectEMaGerDataset
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGRegressor

# TODO: Upload OneSubjectEMaGerDataset repo
# TODO: Make dataset in libemg.datasets
# TODO: Verify this script works


def main():
    parser = ArgumentParser()
    parser.add_argument('--window_size', type=int, default=150, help='Window length (samples). Defaults to 150.')
    parser.add_argument('--window_increment', type=int, default=40, help='Window increment (samples). Defaults to 40.')
    parser.add_argument('--feature_set', type=str, choices=FeatureExtractor().get_feature_groups().keys(), help='Feature set to use. Defaults to HTD.')
    args = parser.parse_args()
    print(args)

    # Load data
    odh = OneSubjectEMaGerDataset().prepare_data()

    # Split into train/test reps
    train_odh = odh.isolate_data('reps', [0, 1, 2, 3, 4])
    test_odh = odh.isolate_data('reps', [5])

    # Extract windows
    train_windows, train_metadata = train_odh.parse_windows(args.window_size, args.window_inc)
    test_windows, test_metadata = test_odh.parse_windows(args.window_size, args.window_inc)

    fe = FeatureExtractor()
    om = OfflineMetrics()
    models = ['SVM', 'LR', 'RF']

    # Make training set
    training_set = {
        'training_features': fe.extract_feature_group(args.feature_set, train_windows, array=True),
        'training_labels': train_metadata['labels']
    }

    test_features = fe.extract_feature_group(args.feature_set, test_windows, array=True)
    test_labels = test_metadata['labels']

    results = {metric: [] for metric in ['R2', 'NRMSE', 'MAE']}

    for model in models:
        reg = EMGRegressor(model)

        # Fit and run model
        reg.fit(training_set.copy())
        predictions = reg.run(test_features)

        metrics = om.extract_offline_metrics(results.keys(), test_labels, predictions)
        for metric in metrics:
            results[metric].append(metrics[metric])
    
    fig, axs = plt.subplots(nrows=len(results), layout='constrained')
    for metric, ax in zip(results.keys(), axs):
        ax.bar(models, results[metric])
        ax.set_title(metric)

    plt.show()


if __name__ == '__main__':
    main()
