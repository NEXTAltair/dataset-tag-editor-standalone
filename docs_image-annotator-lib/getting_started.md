# Getting Started with image-annotator-lib

This guide provides a quick start to using the `image-annotator-lib`, covering installation, basic usage for single and multiple models, and understanding the results.

## Installation

```bash
# Install the library
pip install image-annotator-lib

# Or install the development version from source
# Ensure you are in the project root directory containing pyproject.toml
# Using uv (recommended)
uv pip install -e .[dev]
# Or using pip
# pip install -e .[dev]
```

## Basic Usage

### 1. Import Necessary Libraries

```python
import logging
from pathlib import Path
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators

# Configure logging (optional, for more details)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_annotator_lib")
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed logs
```

### 2. Check Available Models

Before annotating, you can see which models are configured and available:

```python
available_models = list_available_annotators()
print("Available models:", available_models)
# Example output: ['wd-v1-4-vit-tagger-v2', 'aesthetic-shadow-v2', 'blip-large-captioning', ...]
```

This function reads the `annotator_config.toml` file and lists the models ready for use.

### 3. Prepare Images

Load the image(s) you want to annotate using PIL (Pillow). The `annotate` function expects a list of PIL Image objects.

```python
try:
    # Replace with the actual path to your image
    image_path = Path("path/to/your/image.jpg")
    img = Image.open(image_path)
    images_to_process = [img] # Process a single image
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()
except Exception as e:
    print(f"Error opening image {image_path}: {e}")
    exit()
```

### 4. Annotate with a Single Model

Choose a model from the available list and pass it to the `annotate` function.

```python
# Select a model (e.g., WD Tagger)
model_name = "wd-v1-4-vit-tagger-v2"
models_to_use = [model_name]

# Check if the selected model is available (optional but recommended)
if model_name not in available_models:
    print(f"Warning: Selected model '{model_name}' is not available.")
    exit()

# Run annotation
print(f"Annotating image with model: {model_name}...")
results = annotate(images_to_process, models_to_use)
print("Annotation complete.")

# Process results
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")
    if not model_results:
        print("  No results for this image.")
        continue

    # Get the result for the specific model
    result_for_model = model_results.get(model_name)

    if result_for_model:
        if result_for_model.get("error"):
            print(f"  Error: {result_for_model['error']}")
        else:
            tags = result_for_model.get('tags', []) # Get the list of tags
            formatted_output = result_for_model.get('formatted_output') # Model-specific formatted output
            print(f"  Tags: {tags}")
            # print(f"  Formatted Output: {formatted_output}") # Uncomment to see detailed output
    else:
        print(f"  No result found for model: {model_name}")
```

### 5. Annotate with Multiple Models

You can process multiple images with multiple models simultaneously.

```python
# Prepare multiple images
try:
    image_path1 = Path("path/to/your/image1.jpg")
    image_path2 = Path("path/to/your/image2.png")
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    multiple_images_to_process = [img1, img2]
except FileNotFoundError as e:
    print(f"Error: Image file not found: {e}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# Select multiple models (e.g., WD Tagger and an Aesthetic Scorer)
models_to_use_multi = ["wd-v1-4-vit-tagger-v2", "aesthetic-shadow-v2"]

# Check availability (optional)
unavailable = [m for m in models_to_use_multi if m not in available_models]
if unavailable:
    print(f"Warning: The following models are not available: {', '.join(unavailable)}")
    models_to_use_multi = [m for m in models_to_use_multi if m in available_models]
    if not models_to_use_multi:
        print("Error: No available models selected.")
        exit()

# Run annotation
print(f"Annotating {len(multiple_images_to_process)} images with models: {', '.join(models_to_use_multi)}...")
multi_results = annotate(multiple_images_to_process, models_to_use_multi)
print("Annotation complete.")

# Process results for multiple models
for phash, model_results in multi_results.items():
    print(f"--- Image (pHash: {phash}) ---")

    # Access results for each model
    for model_name, result in model_results.items():
        print(f"  Model: {model_name}")
        if result.get("error"):
            print(f"    Error: {result['error']}")
        else:
            tags = result.get('tags', [])
            formatted_output = result.get('formatted_output')
            if tags:
                print(f"    Tags (Top 5): {tags[:5]}")
            if formatted_output:
                # Example for aesthetic scorer (might be in formatted_output or tags)
                if isinstance(formatted_output, (float, int)): # Simple score
                     print(f"    Score: {formatted_output:.4f}")
                elif isinstance(formatted_output, dict): # More complex output
                     print(f"    Formatted Output: {formatted_output}")
                 # Check tags for score pattern if not found elsewhere
                elif any(t.startswith("[SCORE]") for t in tags):
                    score_tag = next((t for t in tags if t.startswith("[SCORE]")), None)
                    if score_tag:
                        try:
                            score_value = float(score_tag.split("[SCORE]")[1])
                            print(f"    Score: {score_value:.4f}")
                        except (IndexError, ValueError):
                            print(f"    Could not parse score from tag: {score_tag}")

    print() # Add a blank line between images
```

## Understanding the Results Structure

The `annotate` function returns a dictionary where keys are the **perceptual hash (pHash)** of the input images. This allows you to associate results with the correct image, even if processing order changes or errors occur.

Each pHash key maps to another dictionary where keys are the **model names** used for annotation. The values contain the annotation results for that specific image and model.

```python
{
    "image1_phash": { # Perceptual hash of the first image
        "model1_name": { # Result from the first model
            "tags": ["tagA", "tagB", ...], # List of generated tags/scores
            "formatted_output": {...}, # Model-specific detailed output
            "error": None # None if successful, error message string otherwise
        },
        "model2_name": { # Result from the second model for the first image
            # ... results ...
        }
    },
    "image2_phash": { # Perceptual hash of the second image
        "model1_name": { ... },
        "model2_name": { ... }
    },
    # Special key if pHash calculation fails
    "unknown_image_0": {
        "model1_name": { ... }
    }
}
```

## Error Handling

Errors during model execution (e.g., model loading failure, inference error) are captured within the result dictionary under the `error` key for the specific model and image. The `annotate` function itself will generally not raise exceptions for individual model failures, allowing partial results to be returned.

```python
# Example checking for errors
results = annotate([img], ["non-existent-model"]) # Use a model known to fail
for phash, model_results in results.items():
    for model_name, result in model_results.items():
        if result.get("error"):
            print(f"Error processing image {phash} with model {model_name}: {result['error']}")
```

## Next Steps

- Explore the `annotator_config.toml` file to see available models and their configurations.
- Refer to the [API Reference](../REFERENCE/api.md) for details on the `annotate` function and result types.
- See the [Developer Guide](../HOW_TO_GUIDES/developer_guide.md) (if available) or [How to Add a New Model](../HOW_TO_GUIDES/add_new_model.md) for extending the library.
