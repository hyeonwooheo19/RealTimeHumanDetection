import cv2
import numpy as np
import torch
import argparse
import time
import random
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image

class WebcamHumanSegmenter:
    def __init__(self, camera_id=0, model_name="facebook/mask2former-swin-base-coco-panoptic"):
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Mask2Former processor and model with label_ids_to_fuse=[] to prevent instance fusion
        self.processor = AutoImageProcessor.from_pretrained(model_name, label_ids_to_fuse=[])
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        
        # In COCO panoptic segmentation, person is class_id 0
        self.person_id = 0
        
        # Set up webcam
        self.camera_id = camera_id
        self.cap = None
        
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
    
    def start_camera(self):

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam with ID {self.camera_id}")
            return False
        return True
    
    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def process_frame(self, frame, downsample_factor=1.0):

        # Check if frame is valid
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: None or empty")
            
        # Optionally resize frame for faster processing
        if downsample_factor != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * downsample_factor), int(w * downsample_factor)
            frame_small = cv2.resize(frame, (new_w, new_h))
        else:
            frame_small = frame
        
        # Convert frame to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
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
        
        # Resize segmentation map back to original size if downsampled
        if downsample_factor != 1.0:
            h, w = frame.shape[:2]
            segmentation = cv2.resize(segmentation.astype(np.float32), (w, h), 
                                     interpolation=cv2.INTER_NEAREST).astype(segmentation.dtype)
        
        # Create a mask for all persons
        person_mask = np.zeros_like(segmentation)
        
        # Count number of person instances
        person_segments = []
        for segment in segments_info:
            if segment["label_id"] == self.person_id:
                person_segments.append(segment)
        
        print(f"Found {len(person_segments)} person instances")
        
        # Create outputs
        black_bg = np.zeros_like(frame)  # Black background with only people
        
        # COMPLETELY REVISED APPROACH FOR OVERLAY
        # Make a copy of the original frame for overlay
        overlay = frame.copy()
        
        # Process each person segment for black background and segmented view
        for i, segment in enumerate(person_segments):
            segment_id = segment["id"]
            color_idx = i % len(self.color_palette)
            color = self.color_palette[color_idx]
            bgr_color = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV
            
            # Create binary mask for this person
            person_instance_mask = (segmentation == segment_id)
            person_mask[person_instance_mask] = 1
            
            # Create 3-channel mask
            mask_3ch = np.zeros_like(frame, dtype=np.uint8)
            mask_3ch[:,:,0] = person_instance_mask.astype(np.uint8) * 255
            mask_3ch[:,:,1] = person_instance_mask.astype(np.uint8) * 255
            mask_3ch[:,:,2] = person_instance_mask.astype(np.uint8) * 255
            
            # Add this person to the black background view
            person_pixels = cv2.bitwise_and(frame, mask_3ch)
            black_bg = cv2.add(black_bg, person_pixels)
            
            # For overlay view - apply semi-transparent color
            # 1. Create a colored mask
            color_overlay = np.zeros_like(frame, dtype=np.uint8)
            color_overlay[:] = bgr_color
            
            # 2. Apply the mask to the color
            color_overlay_masked = cv2.bitwise_and(color_overlay, mask_3ch)
            
            # 3. Get the person pixels
            person_area = cv2.bitwise_and(frame, mask_3ch)
            
            # 4. Create an inverted mask to get everything except the person
            inverted_mask = cv2.bitwise_not(mask_3ch)
            
            # 5. Get the area without the person
            non_person_area = cv2.bitwise_and(overlay, inverted_mask)
            
            # 6. Blend color with person
            alpha = 0.5  # Transparency
            colored_person = cv2.addWeighted(color_overlay_masked, alpha, person_area, 1-alpha, 0)
            
            # 7. Combine the colored person with the rest of the image
            overlay = cv2.add(non_person_area, colored_person)
            
            # Add label to overlay and black background
            y_indices, x_indices = np.where(person_instance_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                top = np.min(y_indices)
                left = np.min(x_indices)
                
                label = f"Person {i+1}"
                cv2.putText(overlay, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                cv2.putText(black_bg, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
        
        # Create segmented view (solid colors)
        segmented = np.zeros_like(frame)
        
        # Process each person again for segmented view
        for i, segment in enumerate(person_segments):
            segment_id = segment["id"]
            color_idx = i % len(self.color_palette)
            color = self.color_palette[color_idx]
            bgr_color = (color[2], color[1], color[0])
            
            # Create mask for this person
            person_instance_mask = (segmentation == segment_id)
            
            # Fill the segmented view with solid color
            segmented_person = np.zeros_like(frame, dtype=np.uint8)
            segmented_person[:] = bgr_color
            mask_3ch = np.zeros_like(frame, dtype=np.uint8)
            mask_3ch[:,:,0] = person_instance_mask.astype(np.uint8) * 255
            mask_3ch[:,:,1] = person_instance_mask.astype(np.uint8) * 255
            mask_3ch[:,:,2] = person_instance_mask.astype(np.uint8) * 255
            
            segmented_person = cv2.bitwise_and(segmented_person, mask_3ch)
            segmented = cv2.add(segmented, segmented_person)
            
            # Add label to segmented view
            y_indices, x_indices = np.where(person_instance_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                top = np.min(y_indices)
                left = np.min(x_indices)
                
                label = f"Person {i+1}"
                cv2.putText(segmented, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
        
        # Add total count of people to all views
        count_text = f"People detected: {len(person_segments)}"
        cv2.putText(overlay, count_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(segmented, count_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(black_bg, count_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, person_mask, overlay, segmented, black_bg
    
    def run_live(self, display_mode="split", downsample=0.5, output_path=None):
        if not self.start_camera():
            return
        
        # Get webcam properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Fixed FPS for recording
        
        # Create video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if display_mode == "split":
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
            else:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize FPS counter
        fps_counter = 0
        fps_timer = time.time()
        fps_display = 0
        
        print("Starting live segmentation with transparent coloring...")
        print("Press 'q' to quit, 'm' to change display mode")
        print(f"Current display mode: {display_mode}")
        
        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                break
            
            # Process the frame
            t0 = time.time()
            original, mask, overlay, segmented, black_bg = self.process_frame(frame, downsample_factor=downsample)
            process_time = time.time() - t0
            
            # Update FPS counter
            fps_counter += 1
            if fps_counter >= 10:
                current_time = time.time()
                fps_display = fps_counter / (current_time - fps_timer)
                fps_counter = 0
                fps_timer = current_time
            
            # Prepare display based on mode
            if display_mode == "split":
                display_frame = np.hstack([original, overlay])  # Show original and overlay side by side
            elif display_mode == "original":
                display_frame = original
            elif display_mode == "mask":
                # Convert mask to visible format
                mask_visible = np.zeros_like(frame, dtype=np.uint8)
                for c in range(3):
                    mask_visible[:, :, c] = mask * 255
                display_frame = mask_visible
            elif display_mode == "segmented":
                display_frame = segmented
            elif display_mode == "overlay":
                display_frame = overlay
            elif display_mode == "black_bg":
                display_frame = black_bg
            else:
                display_frame = overlay  # Default to overlay if mode is unknown
            
            # Add performance metrics
            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Process: {process_time*1000:.1f}ms", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {display_mode}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow("Human Segmentation", display_frame)
            
            # Write to output video if requested
            if writer is not None:
                writer.write(display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Cycle through display modes
                modes = ["split", "original", "mask", "segmented", "overlay", "black_bg"]
                current_idx = modes.index(display_mode) if display_mode in modes else 0
                display_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Switched to display mode: {display_mode}")
                
                # Update writer dimensions if needed
                if writer is not None:
                    writer.release()
                    if display_mode == "split":
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
                    else:
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Clean up
        self.stop_camera()
        if writer is not None:
            writer.release()
            print(f"Recorded video saved to: {output_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Live human segmentation from webcam using Mask2Former")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--mode", type=str, default="overlay", 
                      choices=["split", "original", "mask", "segmented", "overlay", "black_bg"],
                      help="Display mode")
    parser.add_argument("--downsample", type=float, default=0.5, 
                      help="Downsample factor for faster processing")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    args = parser.parse_args()
    
    # Initialize segmenter and run live segmentation
    segmenter = WebcamHumanSegmenter(camera_id=args.camera)
    segmenter.run_live(display_mode=args.mode, downsample=args.downsample, output_path=args.output)