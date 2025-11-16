import torch
from setup_env import device, IMG_SIZE, IMG_CH
import platform
import subprocess
import shutil
import webbrowser
import threading
import http.server as _http_server
import socketserver
import os

# Helper functions
def validate_model_parameters(model):
    """Count and print the number of trainable parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {num_params} total parameters, "
          f"of which {trainable_params} are trainable.")
    
    # Estimate memory usage 
    param_memory = num_params * 4 / (1024 ** 2)  # assuming 32-bit floats
    grad_memory = trainable_params * 4 / (1024 ** 2)  # gradients
    buffer_memory = param_memory*2  # optimizer buffers

    print(f"Estimated memory usage: {param_memory:.2f} MB (parameters), "
          f"{grad_memory:.2f} MB (gradients), {buffer_memory:.2f} MB (buffers)")

def enable_cpu_offload(model):
    """Enable CPU offloading for the model."""
    try:
        model.to("cpu")
        print("Model successfully offloaded to CPU.")
    except Exception as e:
        print(f"Failed to offload model to CPU: {e}")

def generate_image(pipeline, prompt, num_inference_steps=50):
    """Generate an image from a text prompt using the diffusion pipeline."""
    with torch.no_grad():
        result = pipeline(prompt, num_inference_steps=num_inference_steps)
        return result.images[0]

def save_image(image, path):
    """
    Saves the given image to the specified path.

    Args:
        image (PIL.Image.Image): The image to save.
        path (str): The path to save the image.
    """
    try:
        image.save(path)
    except Exception:
        pass

def _serve_file_via_http(path: str, port: int = 8000):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    prev_cwd = os.getcwd()
    try:
        os.chdir(dirpath)
        handler = _http_server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, thread, f"http://127.0.0.1:{port}/{os.path.basename(path)}"
    finally:
        os.chdir(prev_cwd)

def open_image_devcontainer(path: str, port: int = 8000):
    """
    Cross-platform / devcontainer-friendly image opener.
    Order:
    1) xdg-open (typical Linux desktop)
    2) webbrowser.open(file://...)
    3) serve via tiny HTTP server and print URL (use VS Code port forwarding)
    """
    try:
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.run([opener, path], check=False)
            return
    except Exception:
        pass

    try:
        webbrowser.open(f"file://{os.path.abspath(path)}")
        return
    except Exception:
        pass

    try:
        httpd, thread, url = _serve_file_via_http(path, port=port)
        print(f"Started local HTTP server to serve file at: {url}")
        print(f"If this is a devcontainer, forward port {port} and open the URL in your host browser.")
        return
    except Exception as exc:
        print(f"Could not open image automatically: {exc}. Image saved at: {path}")

def benchmark_metrics(model, input_data, device):
    """
    Benchmarks the model's performance on the given input data.
    Handles CUDA out-of-memory errors by falling back to CPU.
    """
    c = None  # Initialize c to avoid unbound errors
    t = None  # Initialize t for time step tensor
    try:
        # Create tensors for class labels and time steps
        c = torch.zeros((input_data.size(0), 1), device=device, dtype=torch.long)
        t = torch.zeros((input_data.size(0),), device=device, dtype=torch.float32)  # Ensure t is 1D
        output = model(input_data, t, c)  # Pass all required arguments
        print("Benchmarking completed successfully.")
    except RuntimeError as e:
        if "CUDA error: out of memory" in str(e):
            print("CUDA out of memory. Falling back to CPU.")
            device = "cpu"
            input_data = input_data.to(device)
            if c is None:
                c = torch.zeros((input_data.size(0), 1), device=device, dtype=torch.long)
            else:
                c = c.to(device)
            if t is None:
                t = torch.zeros((input_data.size(0),), device=device, dtype=torch.float32)  # Ensure t is 1D
            else:
                t = t.to(device)
            output = model(input_data, t, c)
        else:
            raise e
    return output

def log_pipeline_run(prompt, num_inference_steps, generation_time):
    """Functions for logging training progress and metrics."""
    print(f"Prompt: {prompt}")
    print(f"Number of Inference Steps: {num_inference_steps}")
    print(f"Generation Time: {generation_time:.4f} seconds")

def generate_image_with_sdxl(base, refiner, prompt, n_steps=40, high_noise_frac=0.8, device="cuda"):
    """
    Generate an image using SDXL base and refiner pipelines.

    Args:
        base: The base DiffusionPipeline.
        refiner: The refiner DiffusionPipeline.
        prompt: Text prompt for image generation.
        n_steps: Total number of inference steps.
        high_noise_frac: Fraction of steps for the base pipeline.
        device: Device to run the pipelines on (e.g., "cuda").

    Returns:
        A PIL image generated by the refiner pipeline.
    """
    # Debugging: Log the state of refiner before usage
    print("Debug: Verifying refiner state in generate_image_with_sdxl...")
    if refiner is None:
        print("Debug: Refiner is None in generate_image_with_sdxl.")
    else:
        print("Debug: Refiner is valid in generate_image_with_sdxl.")

    # Ensure pipelines are on the correct device
    base.to(device)
    refiner.to(device)

    # Run the base pipeline to generate latents
    latents = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    # Run the refiner pipeline to refine the latents into a final image
    refined_image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
    )
    return refined_image

