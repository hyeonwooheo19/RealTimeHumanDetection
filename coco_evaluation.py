import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from image_segmentation_windows import HumanSegmenter

class SegmentationEvaluator:
    def __init__(self, annotation_file, image_dir):
        """
        Initialize the evaluator with COCO annotations and image directory
        
        Args:
            annotation_file (str): Path to COCO annotation JSON file
            image_dir (str): Directory containing the images
        """
        self.coco_gt = COCO(annotation_file)
        self.image_dir = image_dir
        
        # Filter to only person class (category_id=1 in COCO)
        self.person_cat_id = 1
        self.image_ids = self.coco_gt.getImgIds(catIds=[self.person_cat_id])
        
        # Initialize segmenter
        self.segmenter = HumanSegmenter()
        
        # For storing results
        self.results = []
        self.iou_scores = []
    
    def evaluate(self, num_images=None):
        """
        Evaluate the segmentation model on the dataset
        
        Args:
            num_images (int, optional): Number of images to evaluate (None for all)
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if num_images is not None:
            image_ids = self.image_ids[:num_images]
        else:
            image_ids = self.image_ids
        
        print(f"Evaluating on {len(image_ids)} images...")
        
        # Process each image
        for img_id in tqdm(image_ids):
            self._evaluate_image(img_id)
        
        # If we have no results, return empty metrics
        if not self.results:
            print("No valid results found. Check your annotations and segmentation output.")
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'Mean_IoU': 0.0
            }
        
        try:
            # Calculate AP using COCO API
            coco_dt = self.coco_gt.loadRes(self.results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm')
            coco_eval.params.imgIds = image_ids
            coco_eval.params.catIds = [self.person_cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Get AP values
            ap_metrics = {
                'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
                'AP50': coco_eval.stats[1],  # AP at IoU=0.50
                'AP75': coco_eval.stats[2],  # AP at IoU=0.75
            }
        except Exception as e:
            print(f"Error during COCO evaluation: {e}")
            print("Falling back to custom AP calculation...")
            
            # Custom AP calculation as fallback
            ap_metrics = self._calculate_ap_manually(image_ids)
        
        # Calculate average mask IoU
        mean_iou = np.mean(self.iou_scores) if self.iou_scores else 0
        
        metrics = {
            **ap_metrics,
            'Mean_IoU': mean_iou
        }
        
        return metrics
    
    def _evaluate_image(self, img_id):
        """
        Evaluate a single image
        
        Args:
            img_id (int): COCO image ID
        """
        # Get image info and annotations
        img_info = self.coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return
        
        # Get ground truth annotations for this image
        ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
        annotations = self.coco_gt.loadAnns(ann_ids)
        
        # Check if we have valid annotations
        valid_annotations = []
        for ann in annotations:
            try:
                # Check if annotation can be converted to mask
                if 'segmentation' in ann and ann['segmentation']:
                    # Try to get a mask - this will fail if the segmentation is invalid
                    test_mask = self.coco_gt.annToMask(ann)
                    valid_annotations.append(ann)
            except Exception as e:
                print(f"Skipping invalid annotation (ID: {ann.get('id', 'unknown')}): {e}")
        
        # Skip image if no valid annotations
        if not valid_annotations:
            print(f"No valid annotations for image ID {img_id}")
            return
            
        # Process image with segmenter
        try:
            _, person_mask, _ = self.segmenter.segment_image(img_path, show=False)
            
            # Convert to binary segmentation mask if not already
            binary_mask = (person_mask > 0).astype(np.uint8)
            
            # Get contours from the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour as a person instance
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 100:  # Skip tiny contours
                    continue
                
                # Create mask for this contour
                instance_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
                cv2.drawContours(instance_mask, [contour], -1, 1, -1)
                
                # Calculate IoU with each ground truth mask
                max_iou = 0
                best_gt_id = -1
                
                for ann in valid_annotations:
                    try:
                        gt_mask = self.coco_gt.annToMask(ann)
                        
                        # Calculate IoU
                        intersection = np.logical_and(instance_mask, gt_mask).sum()
                        union = np.logical_or(instance_mask, gt_mask).sum()
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > max_iou:
                            max_iou = iou
                            best_gt_id = ann['id']
                    except Exception as e:
                        print(f"Error calculating IoU: {e}")
                        continue
                
                # Store IoU score if there was a match
                if max_iou > 0:
                    self.iou_scores.append(max_iou)
                
                # Convert mask to RLE format for COCO API
                rle = maskUtils.encode(np.asfortranarray(instance_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Add result
                result = {
                    'image_id': img_id,
                    'category_id': self.person_cat_id,
                    'segmentation': rle,
                    'score': 1.0  # Since we don't have confidence scores, use 1.0
                }
                self.results.append(result)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    def plot_results(self, metrics, save_path=None):
        """
        Plot evaluation results
        
        Args:
            metrics (dict): Evaluation metrics
            save_path (str, optional): Path to save the plot
        """
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('Segmentation Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results plot saved to {save_path}")
        
        plt.show()


    def _calculate_ap_manually(self, image_ids):
        """
        Calculate Average Precision manually when COCO API fails
        
        Args:
            image_ids (list): List of image IDs to evaluate
            
        Returns:
            dict: Dictionary with AP metrics
        """
        # Setup for AP calculation
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Initialize AP containers
        ap_sum = 0.0
        ap50 = 0.0
        ap75 = 0.0
        
        # For each image, calculate precision/recall
        precisions_by_threshold = {t: [] for t in iou_thresholds}
        recalls_by_threshold = {t: [] for t in iou_thresholds}
        
        for img_id in image_ids:
            # Get ground truth annotations
            ann_ids = self.coco_gt.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
            gt_anns = self.coco_gt.loadAnns(ann_ids)
            
            # Get predictions for this image
            pred_anns = [r for r in self.results if r['image_id'] == img_id]
            
            # Skip if no predictions or no ground truth
            if not gt_anns or not pred_anns:
                continue
                
            # Create binary arrays to track matches
            for iou_threshold in iou_thresholds:
                # Arrays to track matches
                gt_matched = np.zeros(len(gt_anns), dtype=bool)
                pred_matched = np.zeros(len(pred_anns), dtype=bool)
                
                # Calculate IoUs between all pred and gt pairs
                for p_idx, pred in enumerate(pred_anns):
                    pred_mask = maskUtils.decode(pred['segmentation'])
                    
                    for g_idx, gt in enumerate(gt_anns):
                        # Skip if already matched
                        if gt_matched[g_idx]:
                            continue
                            
                        try:
                            # Get GT mask
                            if 'segmentation' in gt:
                                gt_mask = self.coco_gt.annToMask(gt)
                            else:
                                # Skip annotations without segmentation
                                continue
                                
                            # Calculate IoU
                            intersection = np.logical_and(pred_mask, gt_mask).sum()
                            union = np.logical_or(pred_mask, gt_mask).sum()
                            iou = intersection / union if union > 0 else 0
                            
                            # Match if IoU exceeds threshold
                            if iou >= iou_threshold:
                                gt_matched[g_idx] = True
                                pred_matched[p_idx] = True
                                break
                        except Exception as e:
                            # Skip problematic GT annotations
                            print(f"Error processing GT annotation: {e}")
                            continue
                
                # Calculate precision and recall
                tp = pred_matched.sum()
                fp = len(pred_matched) - tp
                fn = len(gt_matched) - gt_matched.sum()
                
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                
                precisions_by_threshold[iou_threshold].append(precision)
                recalls_by_threshold[iou_threshold].append(recall)
        
        # Calculate AP at different IoU thresholds
        ap_values = {}
        for iou_threshold in iou_thresholds:
            if not precisions_by_threshold[iou_threshold]:
                ap_values[iou_threshold] = 0.0
                continue
                
            # Average precision at this threshold
            ap_values[iou_threshold] = np.mean(precisions_by_threshold[iou_threshold])
            
        # Calculate final metrics
        ap_mean = np.mean(list(ap_values.values()))
        ap50 = ap_values[0.5]
        ap75 = ap_values[0.75]
        
        return {
            'AP': ap_mean,
            'AP50': ap50,
            'AP75': ap75
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate human segmentation on COCO dataset")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotation JSON file")
    parser.add_argument("--images", required=True, help="Directory containing the images")
    parser.add_argument("--num-images", type=int, default=None, help="Number of images to evaluate (default: all)")
    parser.add_argument("--output", default="evaluation_results.png", help="Path to save results plot")
    parser.add_argument("--fix-annotations", action="store_true", help="Try to fix problematic annotations")
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = SegmentationEvaluator(args.annotations, args.images)
    metrics = evaluator.evaluate(args.num_images)
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    evaluator.plot_results(metrics, args.output)