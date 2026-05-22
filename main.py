import argparse
import os
from individual import Individual
import torch
from net import Net
from torch_utils import get_test_dataset, get_train_dataset
import matplotlib.pyplot as plt
from vae import ConvVAE, vae_loss
from tqdm import tqdm
import random
import numpy as np
import json
import time
import torch.nn.functional as F
from scipy.stats import gamma


def main():
    
    parser = argparse.ArgumentParser(
        description="Dry run to check if everything works."
    )

    # General arguments
    parser.add_argument(
        "--label", 
        type=int, 
        default=9, 
        help="choose label to filter and mutate")

    # Arguments for mutation finding
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=100, 
        help="number of samples to mutate from filtered dataset")
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed")
    
    parser.add_argument(
        "--time_budget", 
        type=int, 
        default=10.0,
        help="time budget in seconds")

    # Hill climbing specific arguments
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=500,
        help="max hill climb steps")

    parser.add_argument(
        "--base_extent", 
        type=float, 
        default=0.5,
        help="start extent for hill climb")


    # Task selection arguments
    parser.add_argument(
        "--filter_dataset",
        action="store_true",
        help="Filter out images that cannot be processed.",
    )

    parser.add_argument(
        "--find_random_mutations",
        action="store_true",
        help="Find random mutations that fail in the model.",
    )

    parser.add_argument(
        "--find_hill_mutations",
        action="store_true",
        help="Find mutations that fail in the model using hill-climbing.",
    )
    
    # Check functions
    parser.add_argument(
        "--check-conversion",
        action="store_true",
        help="Check image conversion functions.",
    )
    parser.add_argument(
        "--check-model-loading",
        action="store_true",
        help="Load DNN under test and check model prediction.",
    )
    parser.add_argument(
        "--check-vae-loading",
        action="store_true",
        help="Load VAE and plot a batch of 4 reconstructions.",
    )

    args = parser.parse_args()
    check_conversion = args.check_conversion
    check_model_loading = args.check_model_loading
    check_vae_loading = args.check_vae_loading
    filter_dataset = args.filter_dataset
    find_random_mutations = args.find_random_mutations
    find_hill_mutations = args.find_hill_mutations
    label = args.label
    seed = args.seed
    subset_size = args.sample_size
    time_budget = args.time_budget
    max_hill_steps = args.max_steps
    base_extent = args.base_extent

    if filter_dataset:
        create_dataset(label)

    if find_random_mutations:
        create_random_mutations(seed, time_budget, subset_size, label)

    if find_hill_mutations:
        create_hill_climbing_mutations(seed, time_budget, subset_size, label, max_hill_steps, base_extent)

    if check_conversion:
        run_check_conversion()

    if check_model_loading:
        run_check_model_loading()
      
    if check_vae_loading:
        run_check_vae_loading()

# Task 0 Filtering dataset
def create_dataset(chosen_label):
    print(f"Filtering dataset for label {chosen_label}...")

    # load model
    model_path = os.path.join("models", "dnn.pt")
    model = Net()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    # load dataset
    test_dataset = get_test_dataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    passed_samples = []

    for idx, (image_tensor, lable_tensor) in enumerate(tqdm(test_loader, desc="Filtering dataset")):
        label = lable_tensor.item()
        
        # check if label matches choosen_label
        if label != chosen_label:
            continue
        
        # check if model predicts the same label
        with torch.no_grad():
            output = model(image_tensor.to("cpu")).squeeze()
            predicted_label = output.argmax(dim=0).item()

        if predicted_label != label:
            continue
        
        # check if conversion to svg and back works and still predicts the same label
        try:
            individual = Individual(tensor_image=image_tensor)
            _, conversion_tensor = individual.get_image_array_and_tensor_representation()

            with torch.no_grad():
                output = model(conversion_tensor.to("cpu")).squeeze()
                converted_predicted_label = output.argmax(dim=0).item()

            if converted_predicted_label != label:
                continue

            # add the sample if all checks passed
            passed_samples.append((idx, Individual(tensor_image=image_tensor)))

        except Exception as e:
            print("Conversion failed for sample index ", idx, " with error: ", e)
            continue

    # save passed samples
    output_dir = os.path.join("filtered_dataset")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join("filtered_dataset", f"class_{chosen_label}.pt")
    torch.save(passed_samples, save_path)

    print(f"Filtering completed for label {chosen_label}. {len(passed_samples)} samples passed all checks and saved to {save_path}.")

# Helper functions
def _load_filtered_dataset(label, subset_size, seed=None):
    """Load filtered dataset and pick a random subset."""
    filtered_dataset_path = os.path.join("filtered_dataset", f"class_{label}.pt")
    if not os.path.exists(filtered_dataset_path):
        print(f"Filtered dataset for label {label} not found. Please run with --filter_dataset first.")
        return None

    samples = torch.load(filtered_dataset_path, weights_only=False)
    if seed is not None:
        random.seed(seed)
    subset = random.sample(samples, min(subset_size, len(samples)))
    return subset

def _load_model():
    """Load the model from disk."""
    model_path = os.path.join("models", "dnn.pt")
    model = Net()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model

def _setup_directory(label, method, seed):
    """Create output directory for saving mutations."""
    output_dir = os.path.join("mutations", f"class_{label}", method, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _save_mutation(individual, mutated_individual, idx, label, pred_label, seed, time_budget, output_dir, extra_metadata=None):
    """Save mutation images and metadata."""
    file_output_dir = os.path.join(output_dir, f"{len(os.listdir(output_dir))+1}")
    os.makedirs(file_output_dir, exist_ok=True)

    original_file = os.path.join(file_output_dir, f"original_{idx}")
    individual.plot_current_image(filename=original_file)
    individual.plot_svg_representation(filename=original_file)

    mutation_file = os.path.join(file_output_dir, f"mutation_{idx}")
    mutated_individual.plot_current_image(filename=mutation_file)
    mutated_individual.plot_svg_representation(filename=mutation_file)

    metadata = {
        "seed": seed,
        "time": time_budget,
        "index": idx,
        "expected_label": label,
        "predicted_label": pred_label,
        "original_image": original_file + ".png",
        "mutated_image": mutation_file + ".png",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = os.path.join(file_output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def _update_progress_bar(pbar, start_time, time_budget):
    """Update the progress bar and return True if time exceeded."""
    elapsed = time.time() - start_time
    pbar.n = min(elapsed, time_budget)
    pbar.refresh()
    if elapsed >= time_budget:
        pbar.close()
        return True
    return False

def _correct_class_confidence(logits, true_label):
    probs = torch.softmax(logits, dim=0)
    return float(probs[true_label])

# Task 0 Finding random mutations
def create_random_mutations(seed, time_budget, subset_size, label):
    print(f"Finding random mutations for label {label} with seed {seed}, time budget of {time_budget} seconds and subset size {subset_size}...")

    subset = _load_filtered_dataset(label, subset_size, seed)

    model = _load_model()

    missclassifications = []

    output_dir = _setup_directory(label, "random", seed)

    start_time = time.time()
    pbar = tqdm(total=time_budget, unit="s", desc="Finding Mutations")

    while True:
        for idx , individual in subset:
            
            if _update_progress_bar(pbar, start_time, time_budget):
                break
            
            # create mutation
            mutated_individual = individual.mutate()

            # prepare tensor
            mut_arr, mut_tensor = mutated_individual.get_image_array_and_tensor_representation()
            mut_tensor = mut_tensor.to("cpu")

            # check mutation
            with torch.no_grad():
                output = model(mut_tensor).squeeze()
                predicted_label = output.argmax(dim=0).item()

            # if mutation found
            if predicted_label != label:
                missclassifications.append(mutated_individual)
                _save_mutation(individual, mutated_individual, idx, label, predicted_label, seed, time_budget, output_dir)

        if _update_progress_bar(pbar, start_time, time_budget):
            break

    print( f"Found {len(missclassifications)} missclassifications for label {label} with seed {seed} and subset size {subset_size}")

    print( f"Saved to {output_dir}")

    evaluate_diveristy(output_dir, missclassifications)

    check_validity(output_dir, missclassifications)

# Task 1 Finding hill-climbing mutations
def create_hill_climbing_mutations(seed, time_budget, subset_size, label, max_hill_step, base_extent):
    print(f"Finding hill-climbing mutations for label {label} with seed {seed}, time budget of {time_budget} seconds, subset size {subset_size}, hill climbing limit of {max_hill_step} and base extent of {base_extent}...")

    subset = _load_filtered_dataset(label, subset_size, seed)

    model = _load_model()

    missclassifications = []

    output_dir = _setup_directory(label, "hill_climb", seed)

    start_time = time.time()
    pbar = tqdm(total=time_budget, unit="s", desc="Finding Mutations")

    while True:
        for idx , individual in subset:
            
            if _update_progress_bar(pbar, start_time, time_budget):
                break
            
            current = individual

            _, tensor = current.get_image_array_and_tensor_representation()

            # initial scoring
            with torch.no_grad():
                output = model(tensor.to("cpu")).squeeze()
            
            current_confidence = _correct_class_confidence(output, label)

            # hill climb variables
            extent = base_extent
            steps_made = 0

            # hill climb
            for step in range(max_hill_step):
                
                if _update_progress_bar(pbar, start_time, time_budget):
                    break
                
                #mutate
                candidate = current.mutate(extent=extent)
                _, cand_tensor = candidate.get_image_array_and_tensor_representation()

                # score mutation
                with torch.no_grad():
                    cand_output = model(cand_tensor.to("cpu")).squeeze()
                    cand_confidence = _correct_class_confidence(cand_output, label)
                
                # if confidence decreased, accept mutation
                if cand_confidence < current_confidence:
                    steps_made += 1
                    current = candidate
                    current_confidence = cand_confidence
                    extent = base_extent * (0.99 ** steps_made)
                
                pred_label = int(cand_output.argmax().item())

                # mutation found
                if pred_label != label:
                    current = candidate
                    break

            # if mutation found
            if pred_label != label:
                missclassifications.append(candidate)

                _save_mutation(individual, current, idx, label, pred_label, seed, time_budget, output_dir)
                
        if _update_progress_bar(pbar, start_time, time_budget):
            break

    print( f"Found {len(missclassifications)} missclassifications for label {label} with seed {seed} and subset size {subset_size}")

    print( f"Saved to {output_dir}")

    evaluate_diveristy(output_dir, missclassifications)

    check_validity(output_dir, missclassifications)

# Task 2 Evaluating diversity
def evaluate_diveristy(path, missclassifications):

    print("Evaluating diversity of missclassifications...")
    
    boldness_values = []
    discontinuity_values = []

    for ind in missclassifications:
        b = ind.get_boldness_feature_value()
        d = ind.get_discontinuity_feature_value()
        boldness_values.append(b)
        discontinuity_values.append(d)

    if not boldness_values:
        print("No feature values computed; feature map not saved")
        return

    boldness = np.array(boldness_values)
    discontinuity = np.array(discontinuity_values)

    x_bins = np.linspace(discontinuity.min(), discontinuity.max(), 7)
    y_bins = np.linspace(boldness.min(), boldness.max(), 7)

    y_bins = np.arange(32, 278+1, 41)
    x_bins = np.arange(0, 35+1, 5)

    heatmap, xedges, yedges = np.histogram2d(discontinuity, boldness, bins=[x_bins, y_bins])

    plt.figure(figsize=(7,6))
    im = plt.imshow(
        heatmap.T,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='Blues'
    )

    x_tick_pos = (xedges[:-1] + xedges[1:]) / 2
    y_tick_pos = (yedges[:-1] + yedges[1:]) / 2

    plt.xticks(x_tick_pos, [str(int(v)) for v in xedges[:-1]])
    plt.yticks(y_tick_pos, [str(int(v)) for v in yedges[:-1]])

    cbar = plt.colorbar(im)
    cbar.set_label("Number of Individuals")
    cbar.set_ticks(np.arange(0, int(np.max(heatmap))+1, 1))
    plt.xlabel("Discontinuity Feature Bins")
    plt.ylabel("Boldness Feature Bins")
    plt.title("Feature Map")
    output_path = os.path.join(path, "Feature_map.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Feature map to {output_path}")

# Task 3 Checking validity with VAE
def check_validity(path, missclassifications):
    print("Checking validity of missclassifications using VAE...")

    vae, threshold = _load_vae()

    vae.eval()

    n = len(missclassifications)
    if n == 0:
        print("No missclassifications to check.")
        return

    # Prepare figure
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    axes = np.atleast_2d(axes)

    for idx, mut_ind in enumerate(missclassifications):
        # Get tensor from mutated individual
        mut_arr, mut_tensor = mut_ind.get_image_array_and_tensor_representation()
        mut_tensor = _normalize_mnist(mut_tensor)
        mut_tensor = mut_tensor.to("cpu")

        if mut_tensor.dim() == 3:  # [C, H, W]
            mut_tensor = mut_tensor.unsqueeze(0)

        # Run through VAE
        with torch.no_grad():
            recon, mu, logvar = vae(mut_tensor)
            total_loss, _, _ = vae_loss(mut_tensor, recon, mu, logvar)
            is_valid = total_loss.item() <= threshold

        recon_img = recon[0, 0].cpu().numpy()

        # Plot top: input (mutated)
        axes[0, idx].imshow(mut_arr, cmap="gray")
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"Input {idx+1}\nMutation")

        # Plot bottom: reconstruction
        axes[1, idx].imshow(recon_img, cmap="gray")
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"Loss: {total_loss.item():.4f}\nValid: {is_valid}\nReconstruction")

    plt.suptitle(f"VAE Reconstruction Check | Loss threshold: {threshold:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(path, exist_ok=True)
    out_file = os.path.join(path, "vae_summary.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
 
    print(f"Saved VAE summary figure to {out_file}")

# Helper function to load VAE and compute/load threshold
def _load_vae():
    print("Loading VAE and computing/loading threshold...")

    vae_path = os.path.join("models", "vae.pt")
    vae = ConvVAE(latent_dim=20)
    vae.load_state_dict(torch.load(vae_path, map_location="cpu"))
    vae.eval()

    threshold_path = "vae_threshold.json"

    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
            threshold = data["threshold"]
        print(f"Loaded VAE threshold: {threshold:.4f}")

        return vae, threshold
    
    print("Threshold file not found. Computing threshold from MNIST test set...")

    # Load MNIST train set
    train_dataset = get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    per_image_losses = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Computing reconstruction errors"):

            recon, mu, logvar = vae(imgs)

            batch_losses = []
            for i in range(imgs.size(0)):
                total_loss, _, _ = vae_loss(imgs[i:i+1], recon[i:i+1], mu[i:i+1], logvar[i:i+1])
                batch_losses.append(total_loss.item())
            per_image_losses.extend(batch_losses)

    per_image_losses = np.array(per_image_losses)
    
    shape, loc, scale = gamma.fit(per_image_losses, floc=0)
    threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=4)

    plt.figure(figsize=(10, 5))
    plt.hist(per_image_losses, bins=50, density=True, alpha=0.6, edgecolor="black", label="Empirical Loss Distribution")
    x = np.linspace(0, per_image_losses.max(), 500)
    pdf = gamma.pdf(x, shape, loc=0, scale=scale)
    plt.plot(x, pdf, "r-", linewidth=2, label="Gamma Fit")
    plt.axvline(threshold, color="blue", linestyle="--", linewidth=2, label=f"95% threshold = {threshold:.4f}")
    plt.title("VAE Per-image Loss Distribution")
    plt.xlabel("VAE loss per image (reconstruction + KL)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vae_loss_threshold_plot.png", dpi=200)
    plt.close()
    print("Saved VAE loss threshold plot to vae_loss_threshold_plot.png")

    return vae, threshold

def _normalize_mnist(img_tensor):
    mean = 0.1307
    std = 0.3081
    return (img_tensor - mean) / std



# Old check functions
def run_check_conversion():
    test_dataset = get_test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    test_image, test_label = next(iter(test_loader))
    individual = Individual(tensor_image=test_image)
    individual.plot_original_image(
        filename=f"original_image_label_{test_label.item()}"
    )
    individual.plot_svg_representation(
        filename=f"svg_representation_label_{test_label.item()}"
    )

def run_check_model_loading():
    model_path = os.path.join("models", "dnn.pt")
    model = Net()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu")
    )
    model.eval()
    test_dataset = get_test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    tensor_test_image, tensor_test_label = next(iter(test_loader))
    with torch.no_grad():
        # removes batch dimension
        output = model(tensor_test_image).squeeze()
        predicted_label = output.argmax().item()
        confidence = torch.softmax(output, dim=0)[predicted_label].item()
    print(
        f"True label: {tensor_test_label.item()}, Predicted label: {predicted_label}, Confidence: {confidence:.4f}"
    )

def run_check_vae_loading():
    model_path = os.path.join("models", "vae.pt")
    vae_folder_name = "vae_reconstructions"
    os.makedirs(vae_folder_name, exist_ok=True)
    vae = ConvVAE(latent_dim=20)
    vae.load_state_dict(
        torch.load(model_path, weights_only=True, map_location="cpu")
    )
    vae.eval()
    test_dataset = get_test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False
    )
    with torch.no_grad():
        for batch_idx, (tensor_batch, _) in enumerate(test_loader):

            reconstructed_tensor_batch, mu, logvar = vae(tensor_batch)
            total_loss, reconstruction_loss, kl = vae_loss(
                tensor_batch=tensor_batch,
                reconstructed_tensor_batch=reconstructed_tensor_batch,
                mu=mu,
                logvar=logvar,
            )

            total_loss /= tensor_batch.size(0)
            reconstruction_loss /= tensor_batch.size(0)
            kl /= tensor_batch.size(0)

            print(
                f"Batch {batch_idx}: Total loss: {total_loss.item():.4f} Reconstruction Loss: {reconstruction_loss.item():.4f}, KL Divergence: {kl.item():.4f}"
            )

            # Show original and reconstructed
            _, axes = plt.subplots(2, tensor_batch.size(0), figsize=(12, 3))
            for i in range(tensor_batch.size(0)):
                axes[0][i].imshow(tensor_batch[i, 0].cpu(), cmap="gray")
                axes[1][i].imshow(
                    reconstructed_tensor_batch[i, 0].cpu(), cmap="gray"
                )
                axes[0][i].axis("off")
                axes[1][i].axis("off")
            plt.suptitle("Top: Original | Bottom: Reconstructed")
            plt.savefig(
                os.path.join(
                    vae_folder_name, f"batch_{batch_idx}_reconstructions.png"
                )
            )
            plt.close()

            break

if __name__ == "__main__":
    main()
