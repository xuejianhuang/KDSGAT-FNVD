import os
import pickle
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import random
from mpl_toolkits.mplot3d import Axes3D


def load_features_and_labels(data_dir, feature, sample_size=1000):
    """
     Load feature files and label data, and randomly select a specified number of samples.

     :param data_dir: Data directory containing feature and label files
     :param feature: Feature type ('text' or others)
     :param sample_size: Number of samples to randomly select
     :return: Randomly selected features, labels, and video IDs
     """
    # Read the label file (assuming it contains label information for each video)
    with open(os.path.join(data_dir, 'data.json'), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Collect all video features and labels
    all_features = []
    all_labels = []
    all_video_ids = []

    for line in lines:
        item = json.loads(line.strip())
        video_id = item["video_id"]
        if video_id=='7203827889748528430': # Skip video without audio in FakeTT dataset
            break
        all_video_ids.append(video_id)
        all_labels.append(item["annotation"])

        # Load different feature files based on the feature type
        if feature == 'text':
            with open(os.path.join(data_dir, 'text_fea', f"{video_id}.pkl"), "rb") as fr:
                feature_data = pickle.load(fr)
                all_features.append(feature_data[0, :].numpy())  # Extract text feature and convert to NumPy array
        elif feature == 'audio':
            with open(os.path.join(data_dir, 'audio_fea', f"{video_id}.pkl"), "rb") as fr:
                feature_data = pickle.load(fr)
                all_features.append(feature_data.mean(dim=1).numpy()) # Calculate mean audio features
        elif feature == 'visual':
            with open(os.path.join(data_dir, 'keyframes_fea', f"{video_id}.pkl"), "rb") as fr:
                feature_data = pickle.load(fr)
                all_features.append(feature_data.mean(dim=1).numpy())  # Calculate mean visual features
        elif feature == 'user':
            with open(os.path.join(data_dir, 'user_fea', f"{video_id}.pkl"), "rb") as fr:
                feature_data = pickle.load(fr)
                all_features.append(feature_data.numpy())

    # Return the last 'sample_size' features, labels, and video_ids
    features = np.array(all_features[-sample_size:])
    labels = np.array(all_labels[-sample_size:])
    video_ids = all_video_ids[-sample_size:]

    return features, labels, video_ids


def apply_tsne(features, n_components=3):
    """
    Apply t-SNE for dimensionality reduction.

    :param features: Input feature matrix
    :param n_components: Number of dimensions after reduction (default is 3)
    :return: Reduced feature matrix
    """
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_features = tsne.fit_transform(features)
    return reduced_features


def plot_tsne_3d(reduced_features, labels, video_ids, output_path):
    """
    Plot a 3D scatter plot of t-SNE reduced features, colored by labels, and save the image.

    :param reduced_features: Reduced features
    :param labels: Sample labels
    :param video_ids: Video IDs for labeling
    :param output_path: File path to save the image
    """
    # Use LabelEncoder to convert labels into integer form
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Create a 3D plot object
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D scatter plot, using different colors for different categories
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],
                         c=encoded_labels, cmap='coolwarm', alpha=0.7)


    #cbar = plt.colorbar(scatter, ax=ax, label='Category Label')
    #ax.set_title(f't-SNE 3D Visualization: {FEATURE} Features')

    # Add legend elements for category labels
    handles, labels_legend = scatter.legend_elements()
    ax.legend(handles, ['True','Fake'], loc='best')

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Dataset name
    DATA = 'FakeTT'
    DATA_DIR = f"../data/{DATA}/"  # Feature directory
    FEATURE = 'text'  # Feature type, can be 'text', 'audio', 'visual', 'user'

    # Load 1000 sample features and labels
    features, labels, video_ids = load_features_and_labels(DATA_DIR, FEATURE, sample_size=1000)

    # Apply t-SNE to reduce features to 3D
    reduced_features = apply_tsne(features, n_components=3)

    # Define the output image path
    output_image_path = f"{DATA}_{FEATURE}.png"

    # Visualize the reduced features and save the plot
    plot_tsne_3d(reduced_features, labels, video_ids, output_image_path)

    print(f"t-SNE visualization saved as: {output_image_path}")
