# CLIP Open-Vocabulary Detection Setup

## What is CLIP?

**CLIP (Contrastive Language-Image Pre-training)** by OpenAI is a vision-language model that can classify images with ANY text description. Unlike YOLO which is limited to 80 COCO classes, CLIP can detect any object you name.

**Your current setup:**
- ✅ YOLOv4-tiny: Fast object detection (bounding boxes)
- ❌ Limited to COCO classes: bottle, cup, person, etc.

**With CLIP:**
- ✅ YOLOv4-tiny: Fast object detection (bounding boxes)
- ✅ CLIP: Classify boxes as ANY object (mug, smartphone, notebook, etc.)

---

## Installation

### Step 1: Install CLIP Dependencies

Inside your Docker container:

```bash
# Install required packages
pip3 install ftfy regex tqdm

# Install CLIP from GitHub
pip3 install git+https://github.com/openai/CLIP.git
```

**Note:** First run will download the CLIP model (~350MB). Make sure you have internet access.

### Step 2: Install PyTorch (if not already installed)

CLIP requires PyTorch. Check if installed:

```bash
python3 -c "import torch; print(torch.__version__)"
```

If not installed:

```bash
# For CPU-only (Python 3.6 compatible)
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

**Note:** This installs PyTorch 1.10 which is the last version supporting Python 3.6.

### Step 3: Verify Installation

```bash
python3 -c "import clip; print('CLIP installed successfully')"
```

---

## Usage

### Option 1: Enable CLIP in Perception Manager

Update your code to use CLIP:

```python
from tiago_lissi.perception.perception_manager_v2 import PerceptionManager

# Initialize with CLIP enabled
perception = PerceptionManager(use_clip=True)

# Detect ANY objects by name
detections = perception.detect_open_vocabulary(
    object_names=["mug", "smartphone", "book", "laptop"],
    confidence_threshold=0.2
)
```

### Option 2: Test CLIP

Run the test script:

```bash
cd /workspace
python3 test_clip.py
```

This will:
1. Show YOLO detections (COCO classes)
2. Show CLIP detections (open vocabulary)
3. Save visualizations to `/workspace/`

---

## How It Works

### YOLO-only (current):
```
Camera → YOLO → Detections (limited to 80 COCO classes)
```

### YOLO + CLIP (new):
```
Camera → YOLO → Bounding Boxes → CLIP → Classify as ANY object
```

**Example:**

```python
# YOLO detects generic "bottle"
yolo_detections = perception.run_yolo(image)
# Result: [{'class_name': 'bottle', 'bbox': [100, 100, 80, 120]}]

# CLIP can distinguish specific bottle types
clip_detections = perception.detect_open_vocabulary(
    object_names=["water bottle", "wine bottle", "soda bottle"]
)
# Result: [{'class_name': 'water bottle', 'bbox': [100, 100, 80, 120], 'confidence': 0.87}]
```

---

## Integration with Embodied Agent

### Update `embodied_agent.py`

Enable CLIP in perception initialization:

```python
# In EmbodiedAgent.__init__()
# Change this:
self.perception = PerceptionManager()

# To this:
self.perception = PerceptionManager(use_clip=True)
```

### Use in Commands

Now you can detect ANY object:

```bash
> grab the mug
> pick up the smartphone
> find the notebook
> bring me the laptop
```

Even if these objects aren't in YOLO's training data!

---

## Performance

### Speed

| Method | Speed | Use Case |
|--------|-------|----------|
| **YOLO-only** | ~50ms | Fast, limited classes |
| **YOLO + CLIP** | ~200ms | Slower, unlimited classes |

**Recommendation:** Use CLIP when you need open-vocabulary detection. For known COCO classes (bottle, cup, person), YOLO-only is faster.

### Accuracy

- **YOLO:** High precision for trained classes
- **CLIP:** Good zero-shot performance, works best with clear, centered objects

### Memory

- **YOLO:** ~50MB
- **CLIP (ViT-B/32):** ~350MB model + ~150MB runtime
- **Total:** ~600MB (acceptable for robot applications)

---

## Configuration Options

### CLIP Models

In `clip_classifier.py`, you can change the model:

```python
# Faster, less accurate
classifier = CLIPClassifier(model_name="RN50", device="cpu")

# Default (recommended)
classifier = CLIPClassifier(model_name="ViT-B/32", device="cpu")

# Slower, more accurate (not recommended for Python 3.6)
# classifier = CLIPClassifier(model_name="ViT-B/16", device="cpu")
```

### Confidence Threshold

Adjust detection sensitivity:

```python
# More strict (fewer false positives)
detections = perception.detect_open_vocabulary(
    object_names=["bottle"],
    confidence_threshold=0.4  # Higher = stricter
)

# More lenient (more detections)
detections = perception.detect_open_vocabulary(
    object_names=["bottle"],
    confidence_threshold=0.1  # Lower = more lenient
)
```

**Recommended:** 0.2 for general use, 0.3 for critical tasks

---

## Example Use Cases

### 1. Detect Specific Object Variants

```python
# Instead of generic "cup"
detections = perception.detect_open_vocabulary(
    object_names=["coffee mug", "tea cup", "water glass"]
)
```

### 2. Find Objects Not in COCO

```python
# These aren't in COCO dataset
detections = perception.detect_open_vocabulary(
    object_names=["marker", "stapler", "remote control", "keys"]
)
```

### 3. Disambiguate Similar Objects

```python
# User says "grab the bottle"
# Multiple bottles visible - which one?
detections = perception.detect_open_vocabulary(
    object_names=["water bottle", "soda bottle", "wine bottle"]
)
```

---

## Troubleshooting

### Error: "No module named 'clip'"

**Solution:**
```bash
pip3 install git+https://github.com/openai/CLIP.git
```

### Error: "No module named 'torch'"

**Solution:**
```bash
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

### CLIP is slow

**Solutions:**
1. Use smaller model: `RN50` instead of `ViT-B/32`
2. Reduce image resolution in preprocessing
3. Only use CLIP when necessary (fall back to YOLO for known classes)

### Low confidence scores

**Solutions:**
1. Lower threshold: `confidence_threshold=0.15`
2. Use more specific object names: "water bottle" instead of "bottle"
3. Ensure good lighting and clear view of objects

---

## Comparison: Before vs After

### Before (YOLO-only)

```bash
> grab the mug
✗ Failed: No object detected
# YOLO doesn't have "mug" class (only "cup")
```

### After (YOLO + CLIP)

```bash
> grab the mug
[YOLO] Detected 1 cup
[CLIP] Classifying as: mug, cup, glass
[CLIP] Result: mug (confidence: 0.82)
✓ Success: Executed grab_bottle
```

---

## Next Steps

1. **Install CLIP** (see Installation section)
2. **Test CLIP** with `python3 test_clip.py`
3. **Enable in agent** by updating `embodied_agent.py`
4. **Try commands** like "grab the smartphone" or "find the notebook"

---

## Files Modified

- ✅ **clip_classifier.py** (new) - CLIP inference engine
- ✅ **perception_manager.py** (updated) - Added `detect_open_vocabulary()` method
- ✅ **test_clip.py** (new) - Test script

---

## Future Enhancements

- [ ] Cache CLIP embeddings for faster repeated classification
- [ ] Use GPU if available (requires CUDA setup)
- [ ] Ensemble prompts for better accuracy
- [ ] Integration with VLM for semantic object descriptions

---

## References

- CLIP Paper: https://arxiv.org/abs/2103.00020
- CLIP GitHub: https://github.com/openai/CLIP
- PyTorch Installation: https://pytorch.org/get-started/locally/
