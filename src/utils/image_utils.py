# src/utils/image_utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import shutil

def setup_directories():
    """Setup required project directories"""
    directories = [
        "data/train/good",
        "data/test/good", 
        "data/test/defective",
        "data/val/good",
        "data/val/defective",
        "models/saved_models",
        "models/deployment",
        "database/images/normal",
        "database/images/defective",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    print("📁 Project directories created successfully!")

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """Resize image with optional aspect ratio preservation"""
    if not maintain_aspect_ratio:
        return cv2.resize(image, target_size)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas and center the image
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    start_y = (target_h - new_h) // 2
    start_x = (target_w - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    
    return canvas

def preprocess_image_for_inference(image: np.ndarray, 
                                 target_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Preprocess image for inference"""
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR format from OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    processed = resize_image(image, target_size, maintain_aspect_ratio=True)
    
    return processed

def validate_image_directory(directory_path: Path, 
                           required_extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> bool:
    """Validate that directory contains valid images"""
    if not directory_path.exists():
        print(f"❌ Directory does not exist: {directory_path}")
        return False
    
    image_files = []
    for ext in required_extensions:
        image_files.extend(directory_path.glob(f"*{ext}"))
        image_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"❌ No valid images found in: {directory_path}")
        return False
    
    print(f"✅ Found {len(image_files)} valid images in: {directory_path}")
    return True

def create_image_grid(images: List[np.ndarray], 
                     grid_size: Tuple[int, int] = None,
                     image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a grid of images for visualization"""
    n_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        grid_w = int(np.ceil(np.sqrt(n_images)))
        grid_h = int(np.ceil(n_images / grid_w))
    else:
        grid_w, grid_h = grid_size
    
    # Resize all images
    resized_images = []
    for img in images:
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(img, image_size)
        resized_images.append(resized)
    
    # Create grid
    grid = np.zeros((grid_h * image_size[1], grid_w * image_size[0], 3), dtype=np.uint8)
    
    for idx, img in enumerate(resized_images):
        if idx >= grid_w * grid_h:
            break
        
        row = idx // grid_w
        col = idx % grid_w
        
        start_y = row * image_size[1]
        end_y = start_y + image_size[1]
        start_x = col * image_size[0]
        end_x = start_x + image_size[0]
        
        grid[start_y:end_y, start_x:end_x] = img
    
    return grid

# src/utils/csv_handler.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class CSVHandler:
    """Handle CSV database operations for detection results"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            columns = [
                'timestamp',
                'anomaly_score', 
                'is_defective',
                'confidence_threshold',
                'image_shape',
                'num_tiles_processed',
                'detection_status'
            ]
            
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)
            print(f"📊 Initialized new CSV database: {self.csv_path}")
        else:
            print(f"📊 Using existing CSV database: {self.csv_path}")
            
    def add_detection(self, detection_record: Dict):
        """Add a new detection record to the CSV"""
        try:
            # Load existing data
            df = pd.read_csv(self.csv_path)
            
            # Add new record
            new_row = pd.DataFrame([detection_record])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(self.csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Error adding detection to CSV: {e}")
            
    def get_recent_detections(self, hours: int = 24) -> pd.DataFrame:
        """Get recent detections within specified hours"""
        try:
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return df
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter recent records
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_df = df[df['timestamp'] >= cutoff_time]
            
            return recent_df
            
        except Exception as e:
            print(f"❌ Error retrieving recent detections: {e}")
            return pd.DataFrame()
            
    def get_statistics(self, days: int = 7) -> Dict:
        """Get detection statistics for specified number of days"""
        try:
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return {
                    'total_detections': 0,
                    'defective_count': 0,
                    'normal_count': 0,
                    'defect_rate': 0.0,
                    'avg_anomaly_score': 0.0,
                    'period_days': days
                }
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by days
            cutoff_time = datetime.now() - timedelta(days=days)
            period_df = df[df['timestamp'] >= cutoff_time]
            
            if period_df.empty:
                return {
                    'total_detections': 0,
                    'defective_count': 0,
                    'normal_count': 0,
                    'defect_rate': 0.0,
                    'avg_anomaly_score': 0.0,
                    'period_days': days
                }
            
            # Calculate statistics
            total_detections = len(period_df)
            defective_count = int(period_df['is_defective'].sum())
            normal_count = total_detections - defective_count
            defect_rate = (defective_count / total_detections) * 100 if total_detections > 0 else 0
            avg_anomaly_score = float(period_df['anomaly_score'].mean())
            
            stats = {
                'total_detections': total_detections,
                'defective_count': defective_count,
                'normal_count': normal_count,
                'defect_rate': defect_rate,
                'avg_anomaly_score': avg_anomaly_score,
                'period_days': days,
                'date_range': {
                    'start': period_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': period_df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            return {'error': str(e)}
            
    def export_report(self, output_path: str, days: int = 30) -> bool:
        """Export detection report for specified period"""
        try:
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                print("⚠️ No data available for report")
                return False
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by days
            cutoff_time = datetime.now() - timedelta(days=days)
            period_df = df[df['timestamp'] >= cutoff_time]
            
            # Save report
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            period_df.to_csv(output_path, index=False)
            
            print(f"📊 Report exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return False

# src/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import pandas as pd

class ResultVisualizer:
    """Visualize detection results and anomaly maps"""
    
    def __init__(self, colormap: str = 'jet'):
        self.colormap = colormap
        
    def create_result_overlay(self, original_image: np.ndarray, 
                            anomaly_map: np.ndarray, 
                            is_defective: bool,
                            alpha: float = 0.5) -> np.ndarray:
        """Create overlay of original image with anomaly map"""
        
        # Resize anomaly map to match original image
        h, w = original_image.shape[:2]
        anomaly_resized = cv2.resize(anomaly_map, (w, h))
        
        # Normalize anomaly map to 0-255
        anomaly_normalized = ((anomaly_resized - anomaly_resized.min()) / 
                            (anomaly_resized.max() - anomaly_resized.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        anomaly_colored = cv2.applyColorMap(anomaly_normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 1-alpha, anomaly_colored, alpha, 0)
        
        # Add status text
        status_text = "DEFECTIVE" if is_defective else "NORMAL"
        status_color = (0, 0, 255) if is_defective else (0, 255, 0)  # Red for defective, Green for normal
        
        # Add text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
        
        # Create background rectangle
        cv2.rectangle(overlay, (10, 10), (20 + text_width, 20 + text_height + baseline), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(overlay, status_text, (15, 15 + text_height), font, font_scale, status_color, thickness)
        
        return overlay
    
    def create_anomaly_heatmap(self, anomaly_map: np.ndarray, 
                             save_path: Optional[str] = None) -> np.ndarray:
        """Create standalone anomaly heatmap"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(anomaly_map, cmap=self.colormap, cbar=True)
        plt.title('Anomaly Heatmap')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            
        # Convert plot to image array
        plt.canvas.draw()
        img_array = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    
    def create_detection_summary(self, results_df: pd.DataFrame, 
                               save_path: Optional[str] = None) -> np.ndarray:
        """Create detection summary visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Detection status distribution
        status_counts = results_df['detection_status'].value_counts()
        axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Detection Status Distribution')
        
        # Anomaly score distribution
        axes[0, 1].hist(results_df['anomaly_score'], bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Anomaly Score Distribution')
        
        # Detection timeline
        if 'timestamp' in results_df.columns:
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            detection_timeline = results_df.groupby(results_df['timestamp'].dt.date).size()
            axes[1, 0].plot(detection_timeline.index, detection_timeline.values, marker='o')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Number of Detections')
            axes[1, 0].set_title('Detection Timeline')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Defect rate by hour (if enough data)
        if len(results_df) > 10 and 'timestamp' in results_df.columns:
            results_df['hour'] = results_df['timestamp'].dt.hour
            hourly_defects = results_df.groupby('hour')['is_defective'].mean() * 100
            axes[1, 1].bar(hourly_defects.index, hourly_defects.values)
            axes[1, 1].set_xlabel('Hour of Day')
            axes[1, 1].set_ylabel('Defect Rate (%)')
            axes[1, 1].set_title('Defect Rate by Hour')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor hourly analysis', 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Hourly Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            
        # Convert plot to image array
        plt.canvas.draw()
        img_array = np.frombuffer(plt.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(plt.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array