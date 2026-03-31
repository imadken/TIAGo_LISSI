#!/usr/bin/env python3
"""
CLIP-based Object Classifier for Open-Vocabulary Detection
Uses CLIP to classify YOLO detections as ANY object name
Compatible with Python 3.6+
"""

import cv2
import numpy as np
import rospy
from typing import List, Dict, Tuple, Optional

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    rospy.logwarn("[CLIP] CLIP not installed. Run: pip3 install git+https://github.com/openai/CLIP.git")


class CLIPClassifier:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize CLIP classifier.

        Args:
            model_name: CLIP model to use ('ViT-B/32', 'ViT-B/16', 'RN50')
            device: 'cuda' or 'cpu'
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available. Install with: pip3 install git+https://github.com/openai/CLIP.git")

        rospy.loginfo("[CLIP] Initializing CLIP classifier...")

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            rospy.logwarn("[CLIP] CUDA not available, using CPU")
            device = "cpu"

        self.device = device
        self.model_name = model_name

        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load(model_name, device=device)
            rospy.loginfo("[CLIP] Loaded model: {} on {}".format(model_name, device))
        except Exception as e:
            rospy.logerr("[CLIP] Failed to load model: {}".format(e))
            raise

        self.model.eval()  # Set to evaluation mode

    def classify_boxes(
        self,
        image: np.ndarray,
        boxes: List[List[int]],
        candidate_labels: List[str],
        confidence_threshold: float = 0.2
    ) -> List[Dict]:
        """
        Classify detected bounding boxes using CLIP.

        Args:
            image: RGB image (H, W, 3)
            boxes: List of bounding boxes [[x, y, w, h], ...]
            candidate_labels: List of object names to classify as
            confidence_threshold: Minimum confidence for classification

        Returns:
            List of dicts: [{'class_name': str, 'confidence': float, 'bbox': [x,y,w,h]}, ...]
        """
        if len(boxes) == 0:
            return []

        if len(candidate_labels) == 0:
            rospy.logwarn("[CLIP] No candidate labels provided")
            return []

        rospy.loginfo("[CLIP] Classifying {} boxes with {} candidate labels".format(
            len(boxes), len(candidate_labels)))

        # Prepare text prompts
        # Use templates to improve zero-shot performance
        text_prompts = [self._create_prompt(label) for label in candidate_labels]

        # Tokenize text
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Process each bounding box
        results = []
        for bbox in boxes:
            x, y, w, h = bbox

            # Crop image to bounding box
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)

            if x2 <= x1 or y2 <= y1:
                rospy.logwarn("[CLIP] Invalid box: {}".format(bbox))
                continue

            cropped = image[y1:y2, x1:x2]

            # Preprocess crop for CLIP
            try:
                # Convert BGR to RGB if needed
                if cropped.shape[2] == 3:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                else:
                    cropped_rgb = cropped

                # CLIP preprocessing expects PIL image
                from PIL import Image
                pil_image = Image.fromarray(cropped_rgb)
                image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

                # Encode image
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # Compute similarity
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(1)

                    confidence = values[0].item()
                    best_idx = indices[0].item()
                    best_label = candidate_labels[best_idx]

                    # Only keep if confidence above threshold
                    if confidence >= confidence_threshold:
                        results.append({
                            'class_name': best_label,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        rospy.loginfo("[CLIP] Box {} -> {} ({:.2f})".format(
                            bbox, best_label, confidence))
                    else:
                        rospy.loginfo("[CLIP] Box {} -> {} ({:.2f}) [below threshold]".format(
                            bbox, best_label, confidence))

            except Exception as e:
                rospy.logerr("[CLIP] Error processing box {}: {}".format(bbox, e))
                continue

        rospy.loginfo("[CLIP] Classified {}/{} boxes".format(len(results), len(boxes)))
        return results

    def classify_image(
        self,
        image: np.ndarray,
        candidate_labels: List[str]
    ) -> Tuple[str, float]:
        """
        Classify entire image with CLIP.

        Args:
            image: RGB image
            candidate_labels: List of possible labels

        Returns:
            (best_label, confidence)
        """
        if len(candidate_labels) == 0:
            return ("unknown", 0.0)

        # Prepare text
        text_prompts = [self._create_prompt(label) for label in candidate_labels]

        try:
            # Convert to PIL
            from PIL import Image
            if image.shape[2] == 3 and image.dtype == np.uint8:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Preprocess
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Encode image and text
                image_features = self.model.encode_image(image_input)
                text_tokens = clip.tokenize(text_prompts).to(self.device)
                text_features = self.model.encode_text(text_tokens)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)

                confidence = values[0].item()
                best_idx = indices[0].item()
                best_label = candidate_labels[best_idx]

                return (best_label, confidence)

        except Exception as e:
            rospy.logerr("[CLIP] Error classifying image: {}".format(e))
            return ("unknown", 0.0)

    def _create_prompt(self, label: str) -> str:
        """
        Create text prompt for CLIP.
        Using templates improves zero-shot performance.

        Args:
            label: Object name

        Returns:
            Formatted prompt
        """
        # Simple template - can be improved with ensemble prompts
        return "a photo of a {}".format(label)


def test_clip():
    """Test CLIP classifier with dummy data."""
    rospy.init_node('clip_test', anonymous=True)

    try:
        # Initialize CLIP (will download model on first run)
        classifier = CLIPClassifier(model_name="ViT-B/32", device="cpu")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test 1: Classify entire image
        labels = ["bottle", "cup", "phone", "book", "person"]
        best_label, confidence = classifier.classify_image(dummy_image, labels)
        print("Image classification: {} ({:.2f})".format(best_label, confidence))

        # Test 2: Classify bounding boxes
        boxes = [
            [100, 100, 80, 120],  # bbox 1
            [300, 200, 100, 100]  # bbox 2
        ]
        results = classifier.classify_boxes(dummy_image, boxes, labels)
        print("Box classifications: {}".format(results))

        rospy.loginfo("[CLIP] Test complete!")

    except Exception as e:
        rospy.logerr("[CLIP] Test failed: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_clip()
