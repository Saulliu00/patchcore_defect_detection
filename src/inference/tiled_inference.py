# src/inference/tiled_inference.py
import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import torchvision.transforms as transforms
from anomalib.models import Patchcore

class TiledInference:
    """Tiled inference for high-resolution images"""
    
    def __init__(self, model_path: str, tile_size: Tuple[int, int] = (256, 256), 
                 stride: Tuple[int, int] = (128, 128), device: str = "cpu"):
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.stride = stride
        self.device = device
        self.model = None
        self.transform = None
        self._load_model()
        self._setup_transform()
        
    def _load_model(self):
        """Load the trained PatchCore model"""
        print(f"🔧 Loading model from {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model with saved config
        model_config = checkpoint['model_config']
        self.model = Patchcore(
            backbone=model_config['backbone'],
            layers=model_config['layers'],
            coreset_sampling_ratio=model_config['coreset_sampling_ratio'],
            num_neighbors=model_config['num_neighbors'],
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store image size and normalization
        self.image_size = tuple(checkpoint['image_size'])
        self.normalization = checkpoint['normalization']
        
    def _setup_transform(self):
        """Setup image preprocessing transforms"""
        if self.normalization == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.tile_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def _extract_tiles(self, image: np.ndarray) -> List[Dict]:
        """Extract tiles from high-resolution image"""
        h, w = image.shape[:2]
        tiles = []
        
        # Calculate number of tiles
        n_tiles_h = (h - self.tile_size[0]) // self.stride[0] + 1
        n_tiles_w = (w - self.tile_size[1]) // self.stride[1] + 1
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile coordinates
                start_h = i * self.stride[0]
                end_h = min(start_h + self.tile_size[0], h)
                start_w = j * self.stride[1]
                end_w = min(start_w + self.tile_size[1], w)
                
                # Extract tile
                tile = image[start_h:end_h, start_w:end_w]
                
                # Pad if necessary
                if tile.shape[0] < self.tile_size[0] or tile.shape[1] < self.tile_size[1]:
                    padded_tile = np.zeros((self.tile_size[0], self.tile_size[1], 3), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append({
                    'tile': tile,
                    'coords': (start_h, end_h, start_w, end_w),
                    'tile_id': (i, j)
                })
                
        return tiles
    
    def _stitch_results(self, tile_results: List[Dict], original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Stitch tile results back into full image"""
        h, w = original_shape[:2]
        
        # Initialize result arrays
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        for result in tile_results:
            coords = result['coords']
            start_h, end_h, start_w, end_w = coords
            
            # Get tile anomaly map and resize to match coordinates
            tile_anomaly = result['anomaly_map']
            actual_h = end_h - start_h
            actual_w = end_w - start_w
            
            if tile_anomaly.shape != (actual_h, actual_w):
                tile_anomaly = cv2.resize(tile_anomaly, (actual_w, actual_h))
            
            # Add to result maps
            anomaly_map[start_h:end_h, start_w:end_w] += tile_anomaly
            count_map[start_h:end_h, start_w:end_w] += 1
        
        # Average overlapping regions
        count_map[count_map == 0] = 1  # Avoid division by zero
        anomaly_map = anomaly_map / count_map
        
        return anomaly_map, count_map
    
    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Dict:
        """Perform tiled inference on high-resolution image"""
        original_shape = image.shape
        
        # Extract tiles
        tiles = self._extract_tiles(image)
        print(f"🔍 Processing {len(tiles)} tiles...")
        
        tile_results = []
        
        with torch.no_grad():
            for tile_info in tiles:
                tile = tile_info['tile']
                
                # Preprocess tile
                tile_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                
                # Get prediction
                result = self.model(tile_tensor)
                
                # Extract anomaly map and score
                anomaly_map = result["anomaly_maps"][0].cpu().numpy()
                anomaly_score = result["pred_scores"][0].cpu().item()
                
                # Resize anomaly map to tile size
                anomaly_map = cv2.resize(anomaly_map, self.tile_size)
                
                tile_results.append({
                    'coords': tile_info['coords'],
                    'anomaly_map': anomaly_map,
                    'anomaly_score': anomaly_score,
                    'tile_id': tile_info['tile_id']
                })
        
        # Stitch results
        full_anomaly_map, count_map = self._stitch_results(tile_results, original_shape)
        
        # Calculate overall anomaly score
        overall_score = np.mean(full_anomaly_map)
        is_anomaly = overall_score > threshold
        
        return {
            'anomaly_map': full_anomaly_map,
            'anomaly_score': overall_score,
            'is_anomaly': is_anomaly,
            'threshold': threshold,
            'tile_results': tile_results,
            'count_map': count_map
        }

# src/inference/camera_capture.py
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import time

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("⚠️ Picamera2 not available. Using OpenCV camera.")

class CameraCapture:
    """Camera capture for Raspberry Pi and standard cameras"""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1920, 1080)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        self.is_pi_camera = False
        self._setup_camera()
        
    def _setup_camera(self):
        """Setup camera based on available hardware"""
        if PICAMERA_AVAILABLE:
            try:
                print("📷 Initializing Raspberry Pi Camera...")
                self.camera = Picamera2()
                
                # Configure camera
                camera_config = self.camera.create_still_configuration(
                    main={"size": self.resolution}
                )
                self.camera.configure(camera_config)
                self.camera.start()
                
                self.is_pi_camera = True
                print("✅ Raspberry Pi Camera initialized successfully")
                
                # Wait for camera to warm up
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to initialize Pi Camera: {e}")
                print("📷 Falling back to USB camera...")
                self._setup_usb_camera()
        else:
            self._setup_usb_camera()
            
    def _setup_usb_camera(self):
        """Setup USB camera using OpenCV"""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.is_pi_camera = False
        print("✅ USB Camera initialized successfully")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture a single image"""
        try:
            if self.is_pi_camera:
                # Raspberry Pi Camera
                image = self.camera.capture_array()
                # Convert from RGB to BGR for OpenCV compatibility
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
            else:
                # USB Camera
                ret, frame = self.camera.read()
                if ret:
                    return frame
                else:
                    print("❌ Failed to capture frame")
                    return None
                    
        except Exception as e:
            print(f"❌ Error capturing image: {e}")
            return None
    
    def save_image(self, image: np.ndarray, path: Path) -> bool:
        """Save captured image"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(path), image)
            return success
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            if self.is_pi_camera:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
            print("📷 Camera released")

# src/deployment/raspberry_pi_detector.py
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.tiled_inference import TiledInference
from inference.camera_capture import CameraCapture
from utils.csv_handler import CSVHandler
from utils.visualization import ResultVisualizer

class RaspberryPiDetector:
    """Main detector class for Raspberry Pi deployment"""
    
    def __init__(self, model_path: str, csv_path: str = "database/detection_results.csv", 
                 confidence_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.csv_path = Path(csv_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.inference_engine = None
        self.camera = None
        self.csv_handler = None
        self.visualizer = None
        
        self._setup_components()
        
    def _setup_components(self):
        """Initialize all components"""
        print("🚀 Initializing Raspberry Pi Detector...")
        
        # Setup inference engine
        print("🧠 Loading inference model...")
        self.inference_engine = TiledInference(
            model_path=self.model_path,
            tile_size=(256, 256),
            stride=(128, 128),
            device="cpu"
        )
        
        # Setup camera
        print("📷 Initializing camera...")
        self.camera = CameraCapture(
            camera_id=0,
            resolution=(1920, 1080)
        )
        
        # Setup CSV handler
        print("📊 Initializing database...")
        self.csv_handler = CSVHandler(self.csv_path)
        
        # Setup visualizer
        self.visualizer = ResultVisualizer()
        
        print("✅ All components initialized successfully!")
        
    def detect_single_image(self, image: np.ndarray, save_results: bool = True) -> Dict:
        """Detect defects in a single image"""
        timestamp = datetime.now()
        
        # Run inference
        print("🔍 Running defect detection...")
        results = self.inference_engine.predict(
            image, 
            threshold=self.confidence_threshold
        )
        
        # Create detection record
        detection_record = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'anomaly_score': float(results['anomaly_score']),
            'is_defective': bool(results['is_anomaly']),
            'confidence_threshold': self.confidence_threshold,
            'image_shape': f"{image.shape[0]}x{image.shape[1]}",
            'num_tiles_processed': len(results['tile_results']),
            'detection_status': 'DEFECTIVE' if results['is_anomaly'] else 'NORMAL'
        }
        
        # Save to CSV if requested
        if save_results:
            self.csv_handler.add_detection(detection_record)
            print(f"💾 Detection result saved to database")
        
        # Add visualization
        results['visualized_image'] = self.visualizer.create_result_overlay(
            image, results['anomaly_map'], results['is_anomaly']
        )
        
        results['detection_record'] = detection_record
        
        return results
    
    def continuous_monitoring(self, capture_interval: float = 5.0, 
                            save_images: bool = True, max_detections: int = None):
        """Run continuous defect monitoring"""
        print("🔄 Starting continuous monitoring...")
        print(f"📷 Capture interval: {capture_interval} seconds")
        
        detection_count = 0
        
        try:
            while True:
                # Check if max detections reached
                if max_detections and detection_count >= max_detections:
                    print(f"🏁 Reached maximum detections limit: {max_detections}")
                    break
                
                # Capture image
                image = self.camera.capture_image()
                if image is None:
                    print("❌ Failed to capture image, retrying...")
                    time.sleep(1)
                    continue
                
                # Run detection
                results = self.detect_single_image(image, save_results=True)
                
                # Display results
                self._display_results(results)
                
                # Save image if requested
                if save_images:
                    self._save_detection_image(image, results, detection_count)
                
                detection_count += 1
                
                # Wait before next capture
                time.sleep(capture_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        finally:
            self.cleanup()
    
    def _display_results(self, results: Dict):
        """Display detection results"""
        record = results['detection_record']
        status = record['detection_status']
        score = record['anomaly_score']
        timestamp = record['timestamp']
        
        status_emoji = "🔴" if status == "DEFECTIVE" else "🟢"
        
        print(f"\n{status_emoji} Detection Result [{timestamp}]")
        print(f"   Status: {status}")
        print(f"   Anomaly Score: {score:.4f}")
        print(f"   Threshold: {self.confidence_threshold:.4f}")
        print(f"   Tiles Processed: {record['num_tiles_processed']}")
        
        if status == "DEFECTIVE":
            print("⚠️  DEFECT DETECTED - Manual inspection recommended!")
    
    def _save_detection_image(self, image: np.ndarray, results: Dict, detection_count: int):
        """Save detection image with overlay"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        status = results['detection_record']['detection_status'].lower()
        
        # Create save directory
        save_dir = Path("database/images") / status
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        original_path = save_dir / f"detection_{detection_count:04d}_{timestamp}_original.jpg"
        self.camera.save_image(image, original_path)
        
        # Save visualization
        viz_path = save_dir / f"detection_{detection_count:04d}_{timestamp}_result.jpg"
        self.camera.save_image(results['visualized_image'], viz_path)
        
        print(f"💾 Images saved: {original_path.name}, {viz_path.name}")
    
    def batch_process_directory(self, input_dir: str, output_dir: str = None):
        """Process a directory of images"""
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path / "results"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"📁 Processing {len(image_files)} images from {input_path}")
        
        results_summary = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n🔍 Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                continue
            
            # Run detection
            results = self.detect_single_image(image, save_results=True)
            
            # Save results
            result_name = image_path.stem + "_result.jpg"
            result_path = output_path / result_name
            cv2.imwrite(str(result_path), results['visualized_image'])
            
            # Add to summary
            record = results['detection_record']
            record['image_name'] = image_path.name
            record['result_image'] = result_name
            results_summary.append(record)
            
            self._display_results(results)
        
        # Save batch summary
        summary_df = pd.DataFrame(results_summary)
        summary_path = output_path / "batch_results_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n✅ Batch processing complete!")
        print(f"📊 Summary saved to: {summary_path}")
        print(f"🖼️ Result images saved to: {output_path}")
        
        return results_summary
    
    def get_detection_statistics(self, days: int = 7) -> Dict:
        """Get detection statistics from the database"""
        return self.csv_handler.get_statistics(days=days)
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        print("🧹 Cleanup completed")

# Example usage script for Raspberry Pi
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/deployment/patchcore_deployment.pth"
    CSV_PATH = "database/detection_results.csv"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Initialize detector
    detector = RaspberryPiDetector(
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Run continuous monitoring
    # detector.continuous_monitoring(
    #     capture_interval=5.0,
    #     save_images=True,
    #     max_detections=100
    # )
    
    # Or run batch processing
    # detector.batch_process_directory("path/to/test/images")