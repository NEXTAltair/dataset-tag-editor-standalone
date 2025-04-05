# Developer Guide for image-annotator-lib

This guide provides information for developers working with or extending the `image-annotator-lib`, covering topics such as adding new models, running tests, and configuring logging.

## How to Add a New Model

This section explains the steps to add a new image annotation model (like a Tagger or Scorer) to the `image-annotator-lib`.

### 1. Implement the Model Class

Create a Python class for your new model. This class must follow the library's class hierarchy.

**Class Hierarchy:**

1.  **`BaseAnnotator` (`core/base.py`)**: Abstract base class for all annotators.
2.  **Framework-Specific Base Classes (`core/base.py`)**: Implement common logic for specific ML frameworks (ONNX, Transformers, TensorFlow, CLIP, etc.).
3.  **Concrete Model Classes (`models/`)**: Inherit from a framework-specific base class and implement only the logic specific to the individual model.

**Implementation Steps:**

1.  **Choose the appropriate framework base class**: Based on the model's framework.

    - ONNX models → `ONNXBaseAnnotator`
    - Transformers models → `TransformersBaseAnnotator`
    - TensorFlow models → `TensorflowBaseAnnotator`
    - CLIP models → `ClipBaseAnnotator`
    - Pipeline models (using multiple sub-models) -> `PipelineBaseAnnotator`

2.  **Create the concrete model class**: Create a new Python file within the `src/image_annotator_lib/models/` directory.

3.  **Define the class**:
    - Inherit from the chosen framework base class.
    - Implement the `__init__` method:
      - Call `super().__init__(model_name)` to initialize the parent class.
      - Add any model-specific initialization (e.g., setting thresholds).
    - **Override necessary abstract methods**: The base classes handle most of the workflow (loading, prediction loop, error handling). You typically only need to implement or override methods specific to how your model processes data. Common methods to override include:
      - `_preprocess_images(self, images: list[Image.Image]) -> Any`: Preprocess a list of images into the format the model expects.
      - `_run_inference(self, processed: Any) -> Any`: Run inference using the preprocessed data.
      - `_format_predictions(self, raw_outputs: Any) -> list[Any]`: Format the raw model outputs into a standardized list.
      - `_generate_tags(self, formatted_output: Any) -> list[str]`: Generate the final list of tags (or score strings like `["[SCORE]0.95"]`) from the formatted output. **This is often the main method you need to implement.**

**Example Implementation (ONNX Tagger):**

```python
# src/image_annotator_lib/models/tagger_onnx.py

from ..core.base import ONNXBaseAnnotator
from PIL import Image
from typing import Any, List

class MyNewONNXTagger(ONNXBaseAnnotator):
    def __init__(self, model_name: str):
        # model_name is automatically passed by the factory
        super().__init__(model_name)
        # Add model-specific initialization here, if needed
        self.my_threshold = 0.5

    # You might only need to implement _generate_tags if the base class
    # handles preprocessing, inference, and formatting suitably.
    def _generate_tags(self, formatted_output: Any) -> List[str]:
        # formatted_output comes from the base class's _format_predictions
        # (or your override if you provided one).
        final_tags = []
        # --- Add your custom logic here ---
        # Example: Iterate through formatted_output, apply threshold, get tag names
        # for score, tag_index in formatted_output:
        #     if score > self.my_threshold:
        #         tag_name = self.get_tag_name(tag_index) # Assuming you load tag names in __init__
        #         final_tags.append(tag_name)
        # --- End custom logic --
        return final_tags
```

### 2. Add Entry to Configuration File

To make the new model available, add an entry for it in the configuration file (`config/annotator_config.toml`).

**Configuration Fields:**

- **Section Name (`[model_unique_name]`)**: A unique name to identify the model within the library.
- `class` (Required): The name of your implemented concrete model class (as a string).
- `model_path` (Required): The path or URL to the model file(s) or repository.
- `estimated_size_gb` (Recommended): Approximate size of the model in GB for memory management.
- `device` (Optional): Specify the device (`"cuda"`, `"cpu"`, etc.) for the model. Defaults based on system availability if not set.
- Other model-specific configuration keys can be added and accessed via `self.config` within your class.

**Example Entry:**

```toml
[my-new-onnx-tagger-v1]
class = "MyNewONNXTagger"
model_path = "path/to/your/model.onnx"
estimated_size_gb = 0.5
# Optional device override
# device = "cuda"
# Model-specific config
# tag_file_path = "path/to/tags.txt"
```

### 3. Verify Functionality

After completing the steps above, test if the new model works correctly using the library.

```python
from image_annotator_lib import annotate, list_available_annotators
from PIL import Image

# Check if the new model appears in the list
available = list_available_annotators()
print(available)
assert "my-new-onnx-tagger-v1" in available

# Run a test annotation
img = Image.open("path/to/test/image.jpg")
results = annotate([img], ["my-new-onnx-tagger-v1"]) # Use the unique name from config
print(results)
```

This completes the process of adding a new model. Consider adding unit tests for your new model class as well (see the Testing section below).

---

## How to Run Tests

This section describes how to run the tests for the `image-annotator-lib` project, which uses `pytest` and `pytest-bdd`.

### 1. Preparation

Ensure you have installed the development dependencies. From the project root directory (containing `pyproject.toml`), run:

```bash
# Make sure your virtual environment is activated
# (e.g., source .venv/bin/activate or .venv\Scripts\activate)

# Install the library with development dependencies
uv pip install -e .[dev]
# or: pip install -e .[dev]
```

### 2. Running Tests

Execute tests using the `pytest` command from the project root directory.

#### 2.1. Run All Tests

```bash
pytest
```

This command discovers and runs all tests under the `tests/` directory (and other locations pytest checks by default).

#### 2.2. Run with Verbose Output

Use the `-v` flag for more detailed output, showing individual test function names and results.

```bash
pytest -v
```

#### 2.3. BDD (Gherkin) Formatted Results

For BDD tests using `pytest-bdd`, display results in a Gherkin-like format:

```bash
pytest --gherkin-terminal-reporter
```

This improves readability by showing which scenarios and steps passed or failed.

#### 2.4. Run Tests in Specific Files or Directories

Target specific tests by providing paths:

```bash
# Run tests in a specific file
pytest tests/unit/test_api.py

# Run all tests in a directory
pytest tests/integration/
```

#### 2.5. Run Specific Tests or Scenarios by Name

Use the `-k` option to run tests matching a name expression (substring matching).

```bash
# Run test functions containing 'annotate'
pytest -k annotate

# Run BDD scenarios containing 'model loading'
pytest -k "model loading"
```

(Note: Use quotes if the name expression contains spaces.)

#### 2.6. Measure Test Coverage

If `pytest-cov` is installed (included in `[dev]` dependencies), measure test coverage:

```bash
pytest --cov=src/image_annotator_lib tests/
```

This generates a report showing how much of the code in `src/image_annotator_lib` is covered by tests.

### 3. Running Tests in VSCode (pytest-bdd extension)

If you use VSCode with the "BDD - Cucumber/Gherkin Full Support" extension (`vtenentes.bdd`), you can run BDD scenarios directly from `.feature` files.

- **Run Scenario**: Executes the scenario under the cursor (Command Palette: `BDD: Run Scenario`).
- **Debug Scenario**: Debugs the scenario under the cursor (Command Palette: `BDD: Debug Scenario`).

Refer to the extension's documentation for more details.

---

## How to Configure Logging

This section explains how to configure and use the logger in `image-annotator-lib`. Proper logging helps in debugging and monitoring the library's behavior.

### 1. Basic Logger Usage

The library uses Python's standard `logging` module. It's recommended to get a logger at the module level.

```python
import logging

# Get logger at module level
# Using __name__ makes the logger name the full module path
# (e.g., 'image_annotator_lib.models.tagger_onnx')
logger = logging.getLogger(__name__)

# Example log messages
logger.info("This is an info message.")
logger.debug("This is a debug message.")
logger.warning("This is a warning message.")
```

### 2. Logger Setup (`setup_logger`)

The basic configuration (level, format, handlers) for the library's loggers is handled by the `setup_logger` function in `core/utils.py`. This function is typically called internally during library initialization.

**Key Features of `setup_logger`:**

- Gets or creates a logger with the specified name.
- Sets the logging level (default is `logging.INFO`).
- Sets a standard log format (`%(asctime)s - %(name)s - %(levelname)s - %(message)s`).
- Configures handlers to output logs to both the console (standard output) and a log file (`logs/image_annotator_lib.log`).
  - The `logs/` directory is created automatically if it doesn't exist.
- Prevents duplicate handler setup.

**Excerpt from `core/utils.py`:**

```python
# src/image_annotator_lib/core/utils.py

import logging
from pathlib import Path

LOG_FILE = Path("logs/image_annotator_lib.log") # Log file path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    # Ensures logger level is set appropriately, even if already configured
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)

    # Check if handlers are already configured for this logger to avoid duplication
    if not logger.handlers:
        # Also check the root logger if propagating
        # This basic setup assumes direct handling, not relying on root logger config
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # Set level for the handler itself if needed, otherwise inherits logger level
        # stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        # File handler
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8") # Use UTF-8
            file_handler.setFormatter(formatter)
            # file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except OSError as e:
            # Handle potential errors during log file setup (e.g., permission denied)
            logger.error(f"Could not set up log file handler at {LOG_FILE}: {e}", exc_info=False)
            # Continue without file logging if setup fails

    # Ensure propagation is handled as intended (usually True by default)
    # logger.propagate = True

    return logger
```

### 3. Controlling Log Levels

You can adjust the verbosity of the logs by changing the log level. Common levels include:

- **`logging.DEBUG`**: Detailed information, useful for diagnosing problems.
- **`logging.INFO`**: Confirmation that things are working as expected.
- **`logging.WARNING`**: Indication of something unexpected or problematic in the near future (e.g., 'disk space low').
- **`logging.ERROR`**: Due to a more serious problem, the software has not been able to perform some function.
- **`logging.CRITICAL`**: A serious error, indicating that the program itself may be unable to continue running.

To change the log level for the entire library or specific modules when using the library, use the standard `logging` module:

```python
import logging

# Example: Set the root logger for 'image_annotator_lib' to DEBUG
logging.getLogger('image_annotator_lib').setLevel(logging.DEBUG)

# Example: Set a specific model module's logger to DEBUG
logging.getLogger('image_annotator_lib.models.tagger_onnx').setLevel(logging.DEBUG)

# You might also need to adjust handler levels if they were set explicitly
# Example: Find and adjust the console handler level for the library's root logger
lib_logger = logging.getLogger('image_annotator_lib')
for handler in lib_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.DEBUG)
```

### 4. Logging Within Classes

When logging within classes, you can use the module-level logger or get a logger specific to the class name for finer granularity.

```python
import logging

logger = logging.getLogger(__name__) # Module-level logger

class MyAnnotator:
    def __init__(self, model_name: str):
        # Option 1: Use module logger
        logger.info(f"Initializing MyAnnotator for {model_name}")

        # Option 2: Use a class-specific logger (optional)
        # self.class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # self.class_logger.info(f"Initializing MyAnnotator instance for {model_name}")

        self.model_name = model_name

    def some_method(self):
        logger.debug(f"Executing some_method in MyAnnotator ({self.model_name})")
        # self.class_logger.debug("Executing some_method")
        try:
            # Perform some operation
            pass
        except Exception as e:
            # Log errors with traceback information
            logger.error(f"Error in some_method: {e}", exc_info=True)
```

### Summary

- Logs are output using the standard `logging` module.
- Basic setup via `core/utils.py:setup_logger` outputs to console and `logs/image_annotator_lib.log`.
- Log levels can be controlled externally using standard `logging` functions.
- Use `logging.getLogger(__name__)` for effective log source tracking.
