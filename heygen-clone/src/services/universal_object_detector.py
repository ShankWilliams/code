# src/services/universal_object_detector.py
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import mediapipe as mp
from segment_anything import SamPredictor, sam_model_registry
from typing import Dict, List, Optional, Any, Tuple
import json

class UniversalObjectDetector:
    """Advanced object detection service for any object type"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLO for general object detection
        self.yolo_model = YOLO('yolov8x.pt')  # Most accurate model
        
        # Initialize MediaPipe for human pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )
        
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7
        )
        
        self.holistic_detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )
        
        # Initialize Segment Anything Model (SAM) for precise segmentation
        try:
            sam_checkpoint = "models/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)
            self.sam_available = True
        except:
            self.sam_available = False
        
        # Object categories and their transformations
        self.object_categories = {
            'person': {
                'detection_method': 'holistic',
                'segmentation': True,
                'transformations': ['style_transfer', 'costume_change', 'age_progression', 'gender_swap', 'fantasy_character']
            },
            'face': {
                'detection_method': 'face_mesh',
                'segmentation': True,
                'transformations': ['face_swap', 'makeup', 'expression_change', 'age_progression']
            },
            'car': {
                'detection_method': 'yolo',
                'segmentation': True,
                'transformations': ['model_change', 'color_change', 'damage_simulation', 'futuristic_upgrade']
            },
            'building': {
                'detection_method': 'yolo',
                'segmentation': True,
                'transformations': ['architectural_style', 'time_period', 'destruction_simulation', 'fantasy_castle']
            },
            'animal': {
                'detection_method': 'yolo',
                'segmentation': True,
                'transformations': ['species_change', 'mythical_creature', 'cartoon_style', 'size_modification']
            },
            'clothing': {
                'detection_method': 'pose_segmentation',
                'segmentation': True,
                'transformations': ['outfit_change', 'material_change', 'historical_costume', 'fantasy_armor']
            },
            'background': {
                'detection_method': 'semantic_segmentation',
                'segmentation': True,
                'transformations': ['environment_change', 'weather_effects', 'time_of_day', 'fantasy_world']
            },
            'custom_object': {
                'detection_method': 'sam_prompt',
                'segmentation': True,
                'transformations': ['style_transfer', 'material_change', 'size_modification', 'color_change']
            }
        }
    
    def detect_objects(self, frame: np.ndarray, target_categories: List[str] = None) -> Dict:
        """Detect all specified objects in frame"""
        try:
            if target_categories is None:
                target_categories = ['person', 'face', 'car', 'building', 'animal']
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            detected_objects = {}
            
            for category in target_categories:
                if category not in self.object_categories:
                    continue
                
                detection_method = self.object_categories[category]['detection_method']
                
                if detection_method == 'yolo':
                    objects = self.detect_with_yolo(frame, category)
                elif detection_method == 'holistic':
                    objects = self.detect_with_holistic(rgb_frame, category)
                elif detection_method == 'face_mesh':
                    objects = self.detect_faces_detailed(rgb_frame)
                elif detection_method == 'pose_segmentation':
                    objects = self.detect_clothing_regions(rgb_frame)
                elif detection_method == 'semantic_segmentation':
                    objects = self.detect_background_regions(frame)
                elif detection_method == 'sam_prompt':
                    objects = []  # Will be handled by user prompts
                else:
                    objects = []
                
                if objects:
                    detected_objects[category] = objects
            
            return {
                'objects': detected_objects,
                'frame_info': {
                    'width': width,
                    'height': height,
                    'total_objects': sum(len(objs) for objs in detected_objects.values())
                }
            }
            
        except Exception as e:
            return {
                'objects': {},
                'frame_info': {'width': frame.shape[1], 'height': frame.shape[0], 'total_objects': 0},
                'error': str(e)
            }
    
    def detect_with_yolo(self, frame: np.ndarray, category: str) -> List[Dict]:
        """Detect objects using YOLO"""
        try:
            results = self.yolo_model(frame, verbose=False)
            objects = []
            
            # Map category to YOLO class names
            category_mapping = {
                'car': ['car', 'truck', 'bus', 'motorcycle'],
                'building': ['building'],
                'animal': ['cat', 'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
                'person': ['person']
            }
            
            target_classes = category_mapping.get(category, [category])
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if class_name in target_classes and confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Get segmentation mask if available
                            mask = None
                            if hasattr(result, 'masks') and result.masks is not None:
                                mask = result.masks.data[i].cpu().numpy()
                            
                            objects.append({
                                'id': f'{category}_{i}',
                                'category': category,
                                'class_name': class_name,
                                'bbox': {
                                    'x1': int(x1), 'y1': int(y1),
                                    'x2': int(x2), 'y2': int(y2),
                                    'width': int(x2 - x1),
                                    'height': int(y2 - y1)
                                },
                                'confidence': confidence,
                                'mask': mask,
                                'center': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)}
                            })
            
            return objects
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def detect_with_holistic(self, rgb_frame: np.ndarray, category: str) -> List[Dict]:
        """Detect full body using MediaPipe Holistic"""
        try:
            results = self.holistic_detector.process(rgb_frame)
            objects = []
            
            if results.pose_landmarks:
                # Extract body landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x * rgb_frame.shape[1],
                        'y': landmark.y * rgb_frame.shape[0],
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # Calculate bounding box from landmarks
                visible_landmarks = [lm for lm in landmarks if lm['visibility'] > 0.5]
                if visible_landmarks:
                    x_coords = [lm['x'] for lm in visible_landmarks]
                    y_coords = [lm['y'] for lm in visible_landmarks]
                    
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    # Add padding
                    padding = 0.1
                    width_pad = (x2 - x1) * padding
                    height_pad = (y2 - y1) * padding
                    
                    x1 = max(0, x1 - width_pad)
                    y1 = max(0, y1 - height_pad)
                    x2 = min(rgb_frame.shape[1], x2 + width_pad)
                    y2 = min(rgb_frame.shape[0], y2 + height_pad)
                
                # Get segmentation mask
                mask = None
                if results.segmentation_mask is not None:
                    mask = results.segmentation_mask
                
                objects.append({
                    'id': 'person_0',
                    'category': 'person',
                    'bbox': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    },
                    'confidence': 0.9,
                    'landmarks': {
                        'pose': landmarks,
                        'face': self.extract_face_landmarks(results.face_landmarks, rgb_frame.shape) if results.face_landmarks else None,
                        'left_hand': self.extract_hand_landmarks(results.left_hand_landmarks, rgb_frame.shape) if results.left_hand_landmarks else None,
                        'right_hand': self.extract_hand_landmarks(results.right_hand_landmarks, rgb_frame.shape) if results.right_hand_landmarks else None
                    },
                    'mask': mask,
                    'center': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)}
                })
            
            return objects
            
        except Exception as e:
            print(f"Holistic detection error: {e}")
            return []
    
    def detect_faces_detailed(self, rgb_frame: np.ndarray) -> List[Dict]:
        """Detailed face detection with landmarks"""
        # This would use the existing face detection from the previous implementation
        # Enhanced with additional detail for full face transformation
        pass
    
    def detect_clothing_regions(self, rgb_frame: np.ndarray) -> List[Dict]:
        """Detect clothing regions using pose landmarks"""
        try:
            results = self.pose_detector.process(rgb_frame)
            clothing_regions = []
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width = rgb_frame.shape[:2]
                
                # Define clothing regions based on pose landmarks
                clothing_areas = {
                    'upper_body': {
                        'landmarks': [11, 12, 23, 24],  # Shoulders and hips
                        'name': 'shirt/jacket'
                    },
                    'lower_body': {
                        'landmarks': [23, 24, 25, 26, 27, 28],  # Hips to ankles
                        'name': 'pants/skirt'
                    },
                    'feet': {
                        'landmarks': [29, 30, 31, 32],  # Foot landmarks
                        'name': 'shoes'
                    }
                }
                
                for region_name, region_info in clothing_areas.items():
                    region_landmarks = []
                    for idx in region_info['landmarks']:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            if lm.visibility > 0.5:
                                region_landmarks.append({
                                    'x': lm.x * width,
                                    'y': lm.y * height,
                                    'z': lm.z
                                })
                    
                    if len(region_landmarks) >= 2:
                        # Calculate bounding box for clothing region
                        x_coords = [lm['x'] for lm in region_landmarks]
                        y_coords = [lm['y'] for lm in region_landmarks]
                        
                        x1, x2 = min(x_coords), max(x_coords)
                        y1, y2 = min(y_coords), max(y_coords)
                        
                        # Add padding for clothing
                        padding = 0.2
                        width_pad = (x2 - x1) * padding
                        height_pad = (y2 - y1) * padding
                        
                        x1 = max(0, x1 - width_pad)
                        y1 = max(0, y1 - height_pad)
                        x2 = min(width, x2 + width_pad)
                        y2 = min(height, y2 + height_pad)
                        
                        clothing_regions.append({
                            'id': f'clothing_{region_name}',
                            'category': 'clothing',
                            'subcategory': region_name,
                            'name': region_info['name'],
                            'bbox': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': 0.8,
                            'landmarks': region_landmarks,
                            'center': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)}
                        })
            
            return clothing_regions
            
        except Exception as e:
            print(f"Clothing detection error: {e}")
            return []
    
    def detect_background_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect and segment background regions"""
        try:
            # Simple background detection - everything not person/object
            # In a full implementation, this would use semantic segmentation models
            height, width = frame.shape[:2]
            
            # For now, return the entire frame as background
            background_regions = [{
                'id': 'background_0',
                'category': 'background',
                'bbox': {
                    'x1': 0, 'y1': 0,
                    'x2': width, 'y2': height,
                    'width': width,
                    'height': height
                },
                'confidence': 1.0,
                'center': {'x': width // 2, 'y': height // 2}
            }]
            
            return background_regions
            
        except Exception as e:
            print(f"Background detection error: {e}")
            return []
    
    def segment_object_with_sam(self, frame: np.ndarray, point_prompt: Tuple[int, int] = None, 
                               bbox_prompt: Tuple[int, int, int, int] = None) -> np.ndarray:
        """Use SAM for precise object segmentation"""
        if not self.sam_available:
            return None
        
        try:
            self.sam_predictor.set_image(frame)
            
            input_point = None
            input_label = None
            input_box = None
            
            if point_prompt:
                input_point = np.array([[point_prompt[0], point_prompt[1]]])
                input_label = np.array([1])
            
            if bbox_prompt:
                input_box = np.array([bbox_prompt])
            
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True
            )
            
            # Return the mask with highest score
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx]
            
        except Exception as e:
            print(f"SAM segmentation error: {e}")
            return None
    
    def extract_face_landmarks(self, face_landmarks, frame_shape):
        """Extract face landmarks from MediaPipe results"""
        if not face_landmarks:
            return None
        
        landmarks = []
        height, width = frame_shape[:2]
        
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': landmark.x * width,
                'y': landmark.y * height,
                'z': landmark.z
            })
        
        return landmarks
    
    def extract_hand_landmarks(self, hand_landmarks, frame_shape):
        """Extract hand landmarks from MediaPipe results"""
        if not hand_landmarks:
            return None
        
        landmarks = []
        height, width = frame_shape[:2]
        
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x * width,
                'y': landmark.y * height,
                'z': landmark.z
            })
        
        return landmarks

class UniversalObjectTransformer:
    """AI-powered transformation service for any detected object"""
    
    def __init__(self, veo3_client):
        self.veo3_client = veo3_client
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transformation processors for different object types
        self.transformers = {
            'person': PersonTransformer(veo3_client),
            'face': FaceTransformer(veo3_client),
            'car': VehicleTransformer(veo3_client),
            'building': ArchitectureTransformer(veo3_client),
            'animal': AnimalTransformer(veo3_client),
            'clothing': ClothingTransformer(veo3_client),
            'background': BackgroundTransformer(veo3_client),
            'custom_object': CustomObjectTransformer(veo3_client)
        }
    
    def transform_objects(self, frame: np.ndarray, detected_objects: Dict, 
                         transformation_config: Dict) -> Dict:
        """Apply transformations to all detected objects"""
        try:
            transformed_frame = frame.copy()
            transformation_metadata = {}
            
            for category, objects in detected_objects.items():
                if category not in transformation_config:
                    continue
                
                transformer = self.transformers.get(category)
                if not transformer:
                    continue
                
                category_config = transformation_config[category]
                category_results = []
                
                for obj in objects:
                    try:
                        result = transformer.transform_object(
                            frame, obj, category_config
                        )
                        
                        if result and 'transformed_region' in result:
                            # Apply transformation to frame
                            bbox = obj['bbox']
                            transformed_region = result['transformed_region']
                            
                            # Resize transformed region to match bbox
                            transformed_resized = cv2.resize(
                                transformed_region, 
                                (bbox['width'], bbox['height'])
                            )
                            
                            # Apply blending based on mask if available
                            if 'mask' in result and result['mask'] is not None:
                                mask = cv2.resize(result['mask'], (bbox['width'], bbox['height']))
                                mask = mask.astype(np.float32) / 255.0
                                
                                # Apply mask blending
                                for c in range(3):
                                    transformed_frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2'], c] = (
                                        transformed_frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2'], c] * (1 - mask) +
                                        transformed_resized[:, :, c] * mask
                                    )
                            else:
                                # Direct replacement
                                blend_strength = category_config.get('blend_strength', 0.8)
                                transformed_frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']] = cv2.addWeighted(
                                    transformed_frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']], 1 - blend_strength,
                                    transformed_resized, blend_strength, 0
                                )
                        
                        category_results.append({
                            'object_id': obj['id'],
                            'transformation_applied': category_config.get('transformation_type', 'unknown'),
                            'confidence': result.get('confidence', 0.8),
                            'processing_time': result.get('processing_time', 0.1)
                        })
                        
                    except Exception as e:
                        print(f"Error transforming object {obj['id']}: {e}")
                        continue
                
                transformation_metadata[category] = category_results
            
            return {
                'transformed_frame': transformed_frame,
                'metadata': transformation_metadata,
                'total_objects_transformed': sum(len(results) for results in transformation_metadata.values())
            }
            
        except Exception as e:
            return {
                'transformed_frame': frame,
                'metadata': {'error': str(e)},
                'total_objects_transformed': 0
            }

# Specialized transformer classes for different object types
class PersonTransformer:
    """Transformer for full person/body modifications"""
    
    def __init__(self, veo3_client):
        self.veo3_client = veo3_client
    
    def transform_object(self, frame: np.ndarray, person_obj: Dict, config: Dict) -> Dict:
        """Transform entire person"""
        try:
            bbox = person_obj['bbox']
            person_region = frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]
            
            transformation_type = config.get('transformation_type', 'style_transfer')
            
            if transformation_type == 'costume_change':
                result = self.veo3_client.change_costume({
                    'person_image': person_region,
                    'costume_style': config.get('costume_style', 'medieval'),
                    'preserve_pose': config.get('preserve_pose', True),
                    'landmarks': person_obj.get('landmarks', {})
                })
            elif transformation_type == 'age_progression':
                result = self.veo3_client.age_person({
                    'person_image': person_region,
                    'target_age': config.get('target_age', 'older'),
                    'age_intensity': config.get('age_intensity', 0.5)
                })
            elif transformation_type == 'fantasy_character':
                result = self.veo3_client.fantasy_transformation({
                    'person_image': person_region,
                    'character_type': config.get('character_type', 'elf'),
                    'transformation_intensity': config.get('intensity', 0.7)
                })
            else:  # Default to style transfer
                result = self.veo3_client.apply_style_transfer({
                    'source_image': person_region,
                    'style_type': config.get('style_type', 'artistic'),
                    'style_strength': config.get('style_strength', 0.7)
                })
            
            return {
                'transformed_region': result.get('transformed_image', person_region),
                'mask': result.get('mask'),
                'confidence': result.get('confidence', 0.8),
                'processing_time': result.get('processing_time', 0.2)
            }
            
        except Exception as e:
            return {'error': str(e)}

class VehicleTransformer:
    """Transformer for cars and vehicles"""
    
    def __init__(self, veo3_client):
        self.veo3_client = veo3_client
    
    def transform_object(self, frame: np.ndarray, vehicle_obj: Dict, config: Dict) -> Dict:
        """Transform vehicle appearance"""
        try:
            bbox = vehicle_obj['bbox']
            vehicle_region = frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]
            
            transformation_type = config.get('transformation_type', 'model_change')
            
            if transformation_type == 'model_change':
                result = self.veo3_client.change_vehicle_model({
                    'vehicle_image': vehicle_region,
                    'target_model': config.get('target_model', 'sports_car'),
                    'preserve_position': True
                })
            elif transformation_type == 'futuristic_upgrade':
                result = self.veo3_client.futuristic_vehicle({
                    'vehicle_image': vehicle_region,
                    'tech_level': config.get('tech_level', 'cyberpunk'),
                    'add_effects': config.get('add_effects', True)
                })
            elif transformation_type == 'damage_simulation':
                result = self.veo3_client.damage_vehicle({
                    'vehicle_image': vehicle_region,
                    'damage_level': config.get('damage_level', 'moderate'),
                    'damage_type': config.get('damage_type', 'crash')
                })
            else:  # Color change
                result = self.veo3_client.change_vehicle_color({
                    'vehicle_image': vehicle_region,
                    'target_color': config.get('target_color', 'red'),
                    'preserve_details': True
                })
            
            return {
                'transformed_region': result.get('transformed_image', vehicle_region),
                'mask': result.get('mask'),
                'confidence': result.get('confidence', 0.8),
                'processing_time': result.get('processing_time', 0.3)
            }
            
        except Exception as e:
            return {'error': str(e)}

class ArchitectureTransformer:
    """Transformer for buildings and architecture"""
    
    def __init__(self, veo3_client):
        self.veo3_client = veo3_client
    
    def transform_object(self, frame: np.ndarray, building_obj: Dict, config: Dict) -> Dict:
        """Transform building appearance"""
        try:
            bbox = building_obj['bbox']
            building_region = frame[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]
            
            transformation_type = config.get('transformation_type', 'architectural_style')
            
            if transformation_type == 'time_period':
                result = self.veo3_client.change_architecture_period({
                    'building_image': building_region,
                    'target_period': config.get('target_period', 'medieval'),
                    'preserve_structure': config.get('preserve_structure', True)
                })
            elif transformation_type == 'fantasy_castle':
                result = self.veo3_client.fantasy_architecture({
                    'building_image': building_region,
                    'fantasy_style': config.get('fantasy_style', 'castle'),
                    'add_magical_effects': config.get('add_effects', True)
                })
            elif transformation_type == 'destruction_simulation':
                result = self.veo3_client.destroy_building({
                    'building_image': building_region,
                    'destruction_level': config.get('destruction_level', 'partial'),
                    'destruction_type': config.get('destruction_type', 'earthquake')
                })
            else:  # Architectural style change
                result = self.veo3_client.change_architectural_style({
                    'building_image': building_region,
                    'target_style': config.get('target_style', 'modern'),
                    'preserve_proportions': True
                })
            
            return {
                'transformed_region': result.get('transformed_image', building_region),
                'mask': result.get('mask'),
                'confidence': result.get('confidence', 0.8),
                'processing_time': result.get('processing_time', 0.4)
            }
            
        except Exception as e:
            return {'error': str(e)}

# Additional transformer classes would follow similar patterns...
class AnimalTransformer:
    """Transformer for animals"""
    pass

class ClothingTransformer:
    """Transformer for clothing items"""
    pass

class BackgroundTransformer:
    """Transformer for background environments"""
    pass

class CustomObjectTransformer:
    """Transformer for user-defined custom objects"""
    pass
