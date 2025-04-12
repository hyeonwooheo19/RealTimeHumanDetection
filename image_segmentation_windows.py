import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageOps
import argparse
import os
import random

class HumanSegmenter:
    def __init__(self, model_name="facebook/mask2former-swin-base-coco-panoptic"):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Mask2Former processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name, label_ids_to_fuse=[])
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        
        # In COCO panoptic segmentation, person is class_id 0
        self.person_id = 0
        
        # Color palette for multiple humans
        self.color_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (165, 42, 42),  # Brown
            (0, 128, 128)   # Teal
        ]
        
    def segment_image(self, image_path, save_path=None, show=True):

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load image using OpenCV first (to ensure proper loading)
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB (PIL format)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_image)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image with Mask2Former
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert prediction to segmentation map
        result = self.processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[image.size[::-1]]
        )[0]
        
        # Get segmentation map and segments info
        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]
        
        # Convert original image to numpy array
        original_image = np.array(image)
        
        # Create transparent colored overlay
        overlay = original_image.copy()
        alpha = 0.5  # Transparency factor
        
        # Create a mask for all persons
        person_mask = np.zeros_like(segmentation)
        
        # Count number of person instances
        person_segments = []
        for segment in segments_info:
            if segment["label_id"] == self.person_id:
                person_segments.append(segment)
        
        # Process each person segment
        for i, segment in enumerate(person_segments):
            segment_id = segment["id"]
            color_idx = i % len(self.color_palette)
            color = self.color_palette[color_idx]
            
            # Create mask for this person
            person_instance_mask = segmentation == segment_id
            person_mask[person_instance_mask] = 1
            
            # Apply color to this person instance
            for c in range(3):
                overlay[:, :, c] = np.where(
                    person_instance_mask,
                    overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                    overlay[:, :, c]
                )
            
            # Get bounding box for this person to place label
            y_indices, x_indices = np.where(person_instance_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                top = np.min(y_indices)
                left = np.min(x_indices)
                
                # Add label
                label = f"Person {i+1}"
                cv2.putText(
                    overlay, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
        
        # Save result if requested
        if save_path:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(person_mask, cmap='gray')
            plt.title("Person Mask")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f"Segmented Persons ({len(person_segments)} detected)")
            plt.axis("off")
            
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Saved result to {save_path}")
            print(f"Detected {len(person_segments)} person(s) in the image")
        
        # Display result if requested
        if show:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(person_mask, cmap='gray')
            plt.title("Person Mask")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title(f"Segmented Persons ({len(person_segments)} detected)")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
        
        return original_image, person_mask, overlay


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Human segmentation in images using Mask2Former")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--output", default=None, help="Path to save output image")
    parser.add_argument("--no-display", action="store_true", help="Disable display of results")
    args = parser.parse_args()
    
    # Initialize segmenter and process image
    segmenter = HumanSegmenter()
    segmenter.segment_image(args.image, args.output, not args.no_display)