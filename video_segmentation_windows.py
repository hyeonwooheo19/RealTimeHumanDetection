import cv2
import numpy as np
import torch
import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
from tqdm import tqdm
import time
import random

class VideoHumanSegmenter:
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
    
    def process_frame(self, frame):
        # Check if frame is valid
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: None or empty")
            
        # Convert frame to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Ensure the image is in RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Process image with Mask2Former
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert prediction to segmentation map
        result = self.processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[pil_image.size[::-1]]
        )[0]
        
        # Get segmentation map and segments info
        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]
        
        # Create mask for persons
        person_mask = np.zeros_like(segmentation)
        for segment in segments_info:
            # Check if the segment is a person
            if segment["label_id"] == self.person_id:
                segment_id = segment["id"]
                person_mask[segmentation == segment_id] = 1
        
        # Create 3-channel mask for visualization
        person_mask_3ch = np.stack([person_mask] * 3, axis=2)
        
        # Create segmented image (only keep pixels where person is detected)
        segmented = np.zeros_like(frame)
        segmented[person_mask_3ch == 1] = frame[person_mask_3ch == 1]
        
        return frame, person_mask, segmented
        
    def segment_video(self, video_path, output_path=None, display=True):

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Process video frames
        pbar = tqdm(total=total_frames, desc="Processing video")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            t0 = time.time()
            original, mask, segmented = self.process_frame(frame)
            process_time = time.time() - t0
            
            # Create side-by-side view
            combined = np.hstack([original, segmented])
            
            # Add frame info
            cv2.putText(combined, f"Frame: {frame_count} | Process time: {process_time:.3f}s", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the result
            if display:
                cv2.imshow("Human Segmentation (Original | Segmented)", combined)
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output video
            if writer is not None:
                writer.write(combined)
            
            frame_count += 1
            pbar.update(1)
        
        # Clean up
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        if output_path:
            print(f"Processed video saved to: {output_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Human segmentation in videos using Mask2Former")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default=None, help="Path to save output video")
    parser.add_argument("--no-display", action="store_true", help="Disable video display during processing")
    args = parser.parse_args()
    
    # Initialize segmenter and process video
    segmenter = VideoHumanSegmenter()
    segmenter.segment_video(args.video, args.output, not args.no_display)