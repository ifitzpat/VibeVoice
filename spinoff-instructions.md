# Creating a Standalone VibeVoice GStreamer Plugin Repository

This guide provides step-by-step instructions for creating a separate repository containing just the GStreamer plugin with VibeVoice as an external dependency.

## Overview

The standalone repository will:
- Contain only the GStreamer plugin code
- Depend on the main VibeVoice package via pip
- Be independently versioned and distributed
- Work as a drop-in GStreamer element

## Repository Structure

Create the following directory structure for your new repository:

```
vibevoice-gstreamer/
├── vibevoice_gstreamer/
│   ├── __init__.py
│   ├── element.py
│   ├── model_manager.py
│   ├── voice_manager.py
│   ├── control_handler.py
│   ├── audio_generator.py
│   └── audio_pusher.py
├── tests/
│   ├── __init__.py
│   ├── test_element.py
│   ├── test_model_manager.py
│   ├── test_voice_manager.py
│   ├── test_control.py
│   ├── test_compilation.py
│   ├── test_audio_generator.py
│   ├── test_audio_pusher.py
│   └── test_plugin.py
├── examples/
│   ├── basic_usage.py
│   └── control_example.py
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
└── setup.py (optional, for older pip versions)
```

## Step-by-Step Setup

### Step 1: Create New Repository

```bash
# Create new directory
mkdir vibevoice-gstreamer
cd vibevoice-gstreamer

# Initialize git
git init

# Create directory structure
mkdir -p vibevoice_gstreamer tests examples
```

### Step 2: Copy Files from VibeVoice Repository

Copy the following files from the VibeVoice repository:

```bash
# From VibeVoice repo root, copy plugin files
cp vibevoice/gstreamer/element.py vibevoice-gstreamer/vibevoice_gstreamer/
cp vibevoice/gstreamer/model_manager.py vibevoice-gstreamer/vibevoice_gstreamer/
cp vibevoice/gstreamer/voice_manager.py vibevoice-gstreamer/vibevoice_gstreamer/
cp vibevoice/gstreamer/control_handler.py vibevoice-gstreamer/vibevoice_gstreamer/
cp vibevoice/gstreamer/audio_generator.py vibevoice-gstreamer/vibevoice_gstreamer/
cp vibevoice/gstreamer/audio_pusher.py vibevoice-gstreamer/vibevoice_gstreamer/

# Copy test files
cp vibevoice/gstreamer/tests/test_*.py vibevoice-gstreamer/tests/

# Copy examples
cp vibevoice/gstreamer/examples/*.py vibevoice-gstreamer/examples/

# Copy documentation (will need modification)
cp vibevoice/gstreamer/README.md vibevoice-gstreamer/
```

### Step 3: Update Import Statements

All files that import from `vibevoice.gstreamer` need to be updated to use `vibevoice_gstreamer`.

#### Files to Update:

**vibevoice_gstreamer/__init__.py** (create new):
```python
"""
VibeVoice GStreamer Plugin

A production-quality GStreamer element for VibeVoice text-to-speech.
"""

__version__ = "1.0.0"

# Import main element for GStreamer plugin discovery
from .element import GstVibeVoice

__all__ = ['GstVibeVoice']
```

**vibevoice_gstreamer/element.py**:
```python
# Change these imports:
from vibevoice.gstreamer.model_manager import ModelManager
from vibevoice.gstreamer.voice_manager import VoiceManager
from vibevoice.gstreamer.control_handler import ControlHandler
from vibevoice.gstreamer.audio_generator import AudioGenerator
from vibevoice.gstreamer.audio_pusher import AudioPusher

# To:
from vibevoice_gstreamer.model_manager import ModelManager
from vibevoice_gstreamer.voice_manager import VoiceManager
from vibevoice_gstreamer.control_handler import ControlHandler
from vibevoice_gstreamer.audio_generator import AudioGenerator
from vibevoice_gstreamer.audio_pusher import AudioPusher

# Also update the plugin registration at the bottom:
__gstelementfactory__ = (
    "vibevoice",  # Element name
    Gst.Rank.NONE,  # Rank
    GstVibeVoice  # Class
)
```

**vibevoice_gstreamer/audio_generator.py**:
```python
# No import changes needed (doesn't import from gstreamer package)
# But ensure it imports from vibevoice main package correctly
```

**vibevoice_gstreamer/audio_pusher.py**:
```python
# No import changes needed
```

**vibevoice_gstreamer/control_handler.py**:
```python
# No import changes needed
```

**vibevoice_gstreamer/model_manager.py**:
```python
# Keep existing imports - these should import from main vibevoice package:
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
```

**vibevoice_gstreamer/voice_manager.py**:
```python
# No changes needed
```

#### Update Test Files:

All test files need import updates:

**tests/test_element.py**:
```python
# Change:
from vibevoice.gstreamer.element import GstVibeVoice

# To:
from vibevoice_gstreamer.element import GstVibeVoice
```

**tests/test_model_manager.py**:
```python
# Change:
from vibevoice.gstreamer.model_manager import ModelManager

# To:
from vibevoice_gstreamer.model_manager import ModelManager
```

**tests/test_voice_manager.py**:
```python
# Change:
from vibevoice.gstreamer.voice_manager import VoiceManager

# To:
from vibevoice_gstreamer.voice_manager import VoiceManager
```

**tests/test_control.py**:
```python
# Change:
from vibevoice.gstreamer.control_handler import ControlHandler

# To:
from vibevoice_gstreamer.control_handler import ControlHandler
```

**tests/test_compilation.py**:
```python
# Change:
from vibevoice.gstreamer.model_manager import ModelManager

# To:
from vibevoice_gstreamer.model_manager import ModelManager
```

**tests/test_audio_generator.py**:
```python
# Change:
from vibevoice.gstreamer.audio_generator import AudioGenerator

# To:
from vibevoice_gstreamer.audio_generator import AudioGenerator
```

**tests/test_audio_pusher.py**:
```python
# Change:
from vibevoice.gstreamer.audio_pusher import AudioPusher

# To:
from vibevoice_gstreamer.audio_pusher import AudioPusher
```

**tests/test_plugin.py**:
```python
# Change:
from vibevoice.gstreamer.element import GstVibeVoice

# To:
from vibevoice_gstreamer.element import GstVibeVoice
```

**tests/__init__.py** (create):
```python
"""Tests for VibeVoice GStreamer plugin"""
```

### Step 4: Update Example Files

**examples/basic_usage.py**:
```python
# No changes needed - examples use the plugin via GStreamer,
# not via Python imports
```

**examples/control_example.py**:
```python
# No changes needed
```

### Step 5: Create Package Configuration

**pyproject.toml**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vibevoice-gstreamer"
version = "1.0.0"
description = "GStreamer plugin for VibeVoice text-to-speech"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["gstreamer", "tts", "text-to-speech", "vibevoice", "speech-synthesis"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
]

dependencies = [
    "vibevoice",  # Main VibeVoice package
    "PyGObject>=3.42.0",  # For GStreamer bindings
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=3.0",
]
cuda = [
    "torch>=2.0.0",  # For GPU acceleration
]

[project.urls]
"Homepage" = "https://github.com/yourusername/vibevoice-gstreamer"
"Bug Tracker" = "https://github.com/yourusername/vibevoice-gstreamer/issues"
"Documentation" = "https://github.com/yourusername/vibevoice-gstreamer#readme"
"Source Code" = "https://github.com/yourusername/vibevoice-gstreamer"

[tool.setuptools]
packages = ["vibevoice_gstreamer"]

[tool.setuptools.package-data]
vibevoice_gstreamer = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```

**setup.py** (optional, for backward compatibility):
```python
#!/usr/bin/env python3
"""Setup script for vibevoice-gstreamer package"""
from setuptools import setup

# Configuration is in pyproject.toml
setup()
```

### Step 6: Create .gitignore

**.gitignore**:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Virtual environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
```

### Step 7: Update README.md

Update the README to reflect the standalone nature:

```markdown
# VibeVoice GStreamer Plugin

[Keep most of the existing README content, but update the installation section]

## Installation

### Install from PyPI (once published)

```bash
pip install vibevoice-gstreamer
```

### Install from Source

```bash
git clone https://github.com/yourusername/vibevoice-gstreamer.git
cd vibevoice-gstreamer
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/vibevoice-gstreamer.git
cd vibevoice-gstreamer
pip install -e ".[dev]"
```

### GPU Support

For CUDA acceleration:

```bash
pip install vibevoice-gstreamer[cuda]
# or
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

[Rest of README remains the same]
```

### Step 8: Create LICENSE File

**LICENSE**:
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Automated Migration Script

To automate the import changes, create this script:

**migrate_imports.sh**:
```bash
#!/bin/bash
# Script to update all import statements

set -e

echo "Updating imports in vibevoice_gstreamer/..."
find vibevoice_gstreamer -name "*.py" -type f -exec sed -i 's/from vibevoice\.gstreamer\./from vibevoice_gstreamer./g' {} \;
find vibevoice_gstreamer -name "*.py" -type f -exec sed -i 's/import vibevoice\.gstreamer\./import vibevoice_gstreamer./g' {} \;

echo "Updating imports in tests/..."
find tests -name "*.py" -type f -exec sed -i 's/from vibevoice\.gstreamer\./from vibevoice_gstreamer./g' {} \;
find tests -name "*.py" -type f -exec sed -i 's/import vibevoice\.gstreamer\./import vibevoice_gstreamer./g' {} \;

echo "Import updates complete!"
```

Make it executable:
```bash
chmod +x migrate_imports.sh
./migrate_imports.sh
```

## Testing the Standalone Package

### Step 1: Create Virtual Environment

```bash
cd vibevoice-gstreamer
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install VibeVoice first
pip install git+https://github.com/original/vibevoice.git

# Or if VibeVoice is on PyPI:
# pip install vibevoice

# Install GStreamer dependencies
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good python3-gi gstreamer1.0-python3-plugin-loader

# Install this package in development mode
pip install -e ".[dev]"
```

### Step 3: Run Tests

```bash
pytest tests/ -v
```

Expected output: 99 tests passing, 12 skipped (if PyTorch not installed)

### Step 4: Verify GStreamer Plugin

```bash
# Check plugin is discoverable
gst-inspect-1.0 vibevoice

# Should show:
# Plugin Details:
#   Name                     vibevoice
#   Description              VibeVoice Text-to-Speech Element
#   ...
```

### Step 5: Test Examples

```bash
python examples/basic_usage.py
python examples/control_example.py
```

## Publishing to PyPI

### Step 1: Build Package

```bash
pip install build twine
python -m build
```

This creates:
- `dist/vibevoice_gstreamer-1.0.0-py3-none-any.whl`
- `dist/vibevoice-gstreamer-1.0.0.tar.gz`

### Step 2: Upload to TestPyPI (optional)

```bash
twine upload --repository testpypi dist/*
```

### Step 3: Upload to PyPI

```bash
twine upload dist/*
```

## GStreamer Plugin Discovery

GStreamer will automatically discover the plugin through Python's plugin loader. The plugin is registered via the `__gstelementfactory__` tuple in `element.py`.

No additional configuration is needed if:
1. The package is installed in the Python environment
2. `gstreamer1.0-python3-plugin-loader` is installed
3. The element module is importable

## Maintenance and Updates

### Syncing with VibeVoice Updates

When the main VibeVoice package is updated, you may need to:

1. Update the dependency version in `pyproject.toml`
2. Test compatibility with new VibeVoice versions
3. Update model_manager.py if the inference API changes

### Version Bumping

Update version in:
- `pyproject.toml` - `version` field
- `vibevoice_gstreamer/__init__.py` - `__version__`

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'vibevoice'`:
```bash
pip install vibevoice
```

### Plugin Not Found

If `gst-inspect-1.0 vibevoice` fails:
```bash
# Ensure Python plugin loader is available
gst-inspect-1.0 python

# Check package is installed
python -c "import vibevoice_gstreamer; print(vibevoice_gstreamer.__file__)"

# Try setting GST_PLUGIN_PATH
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$(python -c "import site; print(site.getsitepackages()[0])")
```

### Test Failures

If tests fail after migration:
1. Verify all imports were updated correctly
2. Check that `vibevoice` is installed
3. Ensure GStreamer development packages are installed

## Summary Checklist

- [ ] Create new repository structure
- [ ] Copy all required files
- [ ] Create `vibevoice_gstreamer/__init__.py`
- [ ] Update imports in all Python files (use migration script)
- [ ] Create `pyproject.toml`
- [ ] Create `.gitignore`
- [ ] Update README.md with new installation instructions
- [ ] Create LICENSE file
- [ ] Create `tests/__init__.py`
- [ ] Test installation in clean virtual environment
- [ ] Run test suite (should have 99 passing)
- [ ] Verify `gst-inspect-1.0 vibevoice` works
- [ ] Test example scripts
- [ ] Initialize git repository
- [ ] Make initial commit
- [ ] Push to GitHub
- [ ] (Optional) Publish to PyPI

## Additional Files Summary

### Files to Copy (8 source files):
1. `element.py`
2. `model_manager.py`
3. `voice_manager.py`
4. `control_handler.py`
5. `audio_generator.py`
6. `audio_pusher.py`
7. All test files (8 files)
8. All example files (2 files)

### Files to Create (6 new files):
1. `vibevoice_gstreamer/__init__.py`
2. `tests/__init__.py`
3. `pyproject.toml`
4. `setup.py` (optional)
5. `.gitignore`
6. `LICENSE`

### Files to Modify (1 file):
1. `README.md` (update installation section)

### Import Changes Required:
- All files: Change `vibevoice.gstreamer` → `vibevoice_gstreamer`
- Keep `vibevoice.modular.*` and `vibevoice.processor.*` unchanged (these are dependencies)

## Contact and Support

For issues specific to the GStreamer plugin, use the new repository's issue tracker.
For VibeVoice model issues, refer to the main VibeVoice repository.
