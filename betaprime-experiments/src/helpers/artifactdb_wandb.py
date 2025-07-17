# Upload the .mat training data to wandb as artifacts

import wandb
import os
# Path to the .mat training files
os.chdir('/Users/rishabhkumar/Kernel-Experiments/betaprime-experiments')
data_dir = os.path.join(os.getcwd(), 'data/001-2014')
print(data_dir)
# List of .mat training files to 
mat_files = [
    "A01T.mat",
    "A02T.mat",
    "A03T.mat",
    "A04T.mat",
    "A05T.mat",
    "A06T.mat",
    "A07T.mat",
    "A08T.mat",
    "A09T.mat"
]

def upload_all_training_files(mat_dir, project="collection-linking-quickstart", entity=None):
    """
    Upload all .mat training files in mat_files list to wandb as artifacts.
    Args:
        mat_dir: Directory containing the .mat files. -> path
        project: wandb project name.
        entity: wandb entity (optional).
    """
    run = wandb.init(project=project)
    for mat_file in mat_files:
        file_path = os.path.join(mat_dir, mat_file)
        artifact_name = f"training-data-{mat_file.replace('.mat','').lower()}"
        artifact_type = "dataset"
        artifact_description = f"Training data file {mat_file} for betaprime experiments"
        upload_artifact(artifact_name, artifact_type, artifact_description, file_path)
    run.finish()

# Example usage:
# upload_all_training_files("/path/to/mat/files")from pathlib import Path

def upload_artifact(artifact_name, artifact_type, artifact_description, artifact_data):
    artifact = wandb.Artifact(artifact_name, type=artifact_type, description=artifact_description)
    artifact.add_file(artifact_data)
    wandb.log_artifact(artifact)

def download_artifact(artifact_name):
    artifact = wandb.use_artifact(artifact_name)
    artifact_dir = artifact.download()

# Example usage
# Load the .mat training data
artifact_name = "betaprime-experiments-training-data"
artifact_type = "dataset"
artifact_description = "Training data for betaprime experiments"
artifact_data = "/Users/rishabhkumar/Kernel-Experiments/betaprime-experiments/data/001-2014/A01T.mat"

# Upload the artifact
upload_all_training_files(data_dir, project="collection-linking-quickstart")

# Download the artifact