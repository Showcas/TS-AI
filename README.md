# TS_AI — Testing and Security of AI Systems

TS_AI is a project focused on testing the robustness, diversity, and validity of AI systems through dataset filtering, mutation-based input generation, and validity analysis. The current implementation centers on MNIST models and supports workflows for generating random and hill-climbing mutations, then analyzing the resulting adversarial or misclassified examples.

## Overview

The project provides a practical framework for evaluating how machine learning models behave under controlled perturbations. It includes tools to filter valid seed inputs, generate mutated samples, assess mutation diversity, and verify whether generated examples remain realistic using a Variational Autoencoder (VAE).

## Features

- Dataset filtering for reliable seed inputs
- Random mutation generation for baseline robustness testing
- Hill-climbing mutation generation for directed adversarial search
- Diversity evaluation based on image features
- Validity checking with a pre-trained VAE
- Docker-based setup for reproducible development

## Project Goals

The main goal of this repository is to study how AI systems respond to modified inputs and to evaluate the quality of those modifications. This includes not only whether a model fails, but also whether the generated inputs are diverse and still represent plausible data.

# 1. Setup

## 1.1 Docker setup

### 1.1.1 Install Docker

Install [Docker](https://docs.docker.com/engine/install/) for your system.

### 1.1.2 Build the container

```bash
docker build -t dockercontainervm/model-based-tig:latest .
```

or download it from the Docker Hub:

```bash
docker pull dockercontainervm/model-based-tig:latest
```

The command will create an image of ~2GB with all the dependencies.

### 1.1.3 Use VSCode Devcontainer

- Download [VSCode](https://code.visualstudio.com/Download) for your platform;
- Install DevContainer Extension;
- In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the "Dev Containers: Open Folder in Container..." command;

The extension will take the `dockercontainervm/model-based-tig:latest` image and create a container to let you conveniently develop within it. Once inside the container, select the interpeter in VSCode located in `/root/.venv/bin/python`.

# 2. Trained models

Download trained models by going on your browser at this [link](https://drive.switch.ch/index.php/s/UxszD6p4ZChS1P3) or on your command line:

```bash
curl -L https://drive.switch.ch/index.php/s/UxszD6p4ZChS1P3/download -o models.zip
```

Then, unzip the directory, and rename it to `models`. The two pre-trained models are:

- ./models/dnn.pt (MNIST classifier, accuracy 99% on the test set)
- ./models/vae.pt (Variational Autoencoder trained on MNIST for validity evaluation)

# 3. Usage 
This project allows filtering datasets, generating random or hill-climbing mutations, and evaluating their diversity and validity. The following explains how to run each task and the reasoning behind the chosen approach. All the results can be found in the mutations folder.

## 1. Dataset Filtering – Filters out images that cannot be reliably processed or predicted by the model.
Purpose: Select only images that are correctly classified and convertible to SVG.

Design Choice: Filtering ensures that subsequent mutations start from clean, reliable examples, avoiding wasting time on misclassified or corrupted images.

Simplest run
```bash
python main.py --filter_dataset
```

All Flags used
```bash
# label chooses the class you want to work with
python main.py --filter_dataset --label 9
```


What happens:
- Loads the full dataset and DNN model.
- Keeps images whose predicted label matches the true label.
- Checks that converting to SVG and back preserves the label.
- Saves the filtered dataset in filtered_dataset/class_<label>.pt.

## 2. Random Mutation Generation – Generates random image mutations to find inputs that the model misclassifies.
Purpose: Generate random modifications of images to find model misclassifications.

Design Choice: Random mutations provide a baseline test of robustness and are simple to implement.

Simplest run
```bash
python main.py --find_random_mutations
```

All Flags used
```bash
# label chooses the class
# sample size the amout of samples of the filtered dataset you want to use
# time budget is the total time you give for mutations to be found
python main.py --find_random_mutations --label 9 --sample_size 100 --time_budget 10
```

What happens:
- Selects a subset of filtered images.
- Applies random changes to the SVG representation of each image (mutate function).
- Checks if the DNN misclassifies the mutated image.
- Saves misclassified images and metadata in mutations/class_<label>/random/seed_<seed>/.


Mutation Details:
- Each line, curve, or arc in the SVG is slightly perturbed by a random offset.
- The extent parameter controls mutation magnitude (larger = more extreme changes).

## 3. Hill-Climbing Mutation Generation
Purpose: Apply a directed search to generate mutations that more efficiently cause misclassification.

Design Choice: Hill-climbing explores mutations that decrease model confidence in the correct label, making it more likely to find difficult adversarial examples.

Simplest run
```bash
python main.py --find_hill_mutations
```

All Flags used
```bash
# label chooses the class
# sample size the amout of samples of the filtered dataset you want to use
# time budget is the total time you give for mutations to be found
# max steps is the maximum number of attemps a hillclimb can make to find a missclassification
# base extent defines the initial mutation severity, which is then later reduced with every hill climb step
python main.py --find_hill_mutations --label 9 --sample_size 100 --time_budget 10 --max_steps 500 --base_extent 0.5 

```

What happens:
- Start with a subset of filtered images.
- For each image:
    - Apply a mutation using the `mutate(extent)` function.
    - Compute the model’s confidence in the correct label.
    - If confidence decreases, accept the mutation (this ensures that is slowly moves away from the predicting the right label towards a missclassification)
    - Gradually reduce the mutation extent to fine-tune subtle changes.
    - Repeat until a misclassification occurs or the maximum steps/time budget is reached.
- Saves misclassified images and metadata in mutations/class_<label>/hill_climb/seed_<seed>/.


## 4. Diversity Evaluation – Computes boldness and discontinuity features of mutated images.
Purpose: Assess how varied the misclassified mutations are.
Design Choice: Measures two features—boldness and discontinuity—and plots a heatmap to visualize coverage of the feature space.

Executed after collecting mutations

What happens:
- Extracts boldness and discontinuity features for each misclassified image.
- Generates a 2D heatmap showing distribution of mutations.
- Saves the heatmap as Feature_map.png in the output directory.

## 5. VAE Validity Check – Ensures that mutated images are valid using a Variational Autoencoder.
Purpose: Ensure mutated images are realistic and not just extreme, unrealistic perturbations.
Design Choice: Uses a trained VAE to measure reconstruction loss; images with low loss are considered valid.

What happens:
- Loads the VAE and computes the reconstruction loss for each mutated image.
- Compares loss against a precomputed threshold (or computes it if missing).
- Saves a visualization of the mutated images and their VAE reconstructions as vae_summary.png.
