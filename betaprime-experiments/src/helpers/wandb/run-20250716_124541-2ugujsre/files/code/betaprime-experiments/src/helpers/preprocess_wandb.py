import sys, time
sys.path.append('/Users/rishabhkumar/Kernel-Experiments/betaprime-experiments/src/helpers/datautils')
import wandb
from datautils import mat_extractor, multi_signal_cov_calculator, single_signal_cov_calculator
import matplotlib.pyplot as plt
import seaborn as sns
# Download the training data from wandb
def preprocess_and_upload_training_data(project="collection-linking-quickstart", dataset_name="training-data-a01t:v0"):
    run = wandb.init(project=project)
    artifact = run.use_artifact(dataset_name)
    artifact_dir = artifact.download()

    # Preprocess the training data
    x_train, y_train = mat_extractor(
        path=artifact_dir + '/A01T.mat',
        beg=500,
        end=1500,
        remove_eog=True,
        bpf_dict={'apply': True, 'fs': 250, 'lc': 4, 'hc': 38, 'order': 5},
        channel_norm=True
    )

    # Upload the preprocessed data to wandb
    wandb.log({"x_train": x_train, "y_train": y_train})
    cov_train = single_signal_cov_calculator(x_train[0])
    plt.figure(figsize=(10, 10))
    sns.heatmap(cov_train, cmap='viridis')
    plt.title('Covariance Matrix of Training Data')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    plt.xticks(range(22), range(22))
    plt.yticks(range(22), range(22))
    plt.tight_layout()
    plt.savefig('cov_train.png')
    wandb.log({"cov_train": wandb.Image('cov_train.png')})

    run.finish()

preprocess_and_upload_training_data()



# Preprocess the training data

# Upload the preprocessed data to wandb

# Download the test data from wandb

# Preprocess the test data

# Upload the preprocessed data to wandb