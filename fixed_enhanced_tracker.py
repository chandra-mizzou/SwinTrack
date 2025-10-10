# Multi-model tracker with m-by-n consensus merging and stable identities
# - Reads images from a folder
# - Runs YOLOv10 (falls back to v8), Faster R-CNN, RetinaNet, RF-DETR
# - m-by-n consensus: a detection is kept if supported by at least M distinct sources (IoU >= CONSENSUS_IOU)
# - Then clusters kept detections by IoU >= CLUSTER_IOU and averages boxes
# - Chooses class labels by source preference: RFDETR > FRCNN > YOLO > Retina
# - Stable names per class (e.g., car_1, car_2...), reuse only if bbox center within 10 px (over up to 3 missed frames)
# - Enhanced BERT processing for complex tracking commands with MULTIPLE TARGET SUPPORT
# - Press 'c' -> "track <description>" (BERT-based selection), "reset", "quit"
# - Tracks only the selected target(s) until lost (missed > 3 frames), then resumes all
# - RETAINS ALL LABELS: Selected targets show bbox+label in GREEN, others show only label in RED

import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
)
from ultralytics import YOLO
from transformers import BertTokenizer, BertModel
from collections import defaultdict, Counter
import re
import math

# Avoid Apple MPS missing-op crash for RF-DETR resizing (safe no-op elsewhere)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

gt_file_path = "/Users/chandra/Documents/Mizzou/Visual_language_tracking/July17_2025/Datasets/UAVDT/groundtruth.txt"

# input_folder = '/Users/chandra/Documents/Mizzou/Visual_language_tracking/Datasets/UCF/actions1_all'
# output_folder = '/Users/chandra/Documents/Mizzou/Visual_language_tracking/July17_2025/DeepSORT/MMH'

input_folder = '/Users/chandra/Documents/Mizzou/Visual_language_tracking/July17_2025/Datasets/basketball'
output_folder = '/Users/chandra/Documents/Mizzou/Visual_language_tracking/July17_2025/Datasets/basketball_op'

# -------- config --------
INPUT_DIR = input_folder
OUTPUT_DIR = output_folder
# -------- config --------

# m-by-n consensus settings
CONSENSUS_SOURCES_REQUIRED = 3   # m: set 1 for 1-of-4, 2 for 2-of-4, etc.
CONSENSUS_IOU = 0.6               # IoU to consider two detections "agreeing" for consensus

# clustering/merging after consensus
CLUSTER_IOU = 0.6                 # IoU to cluster detections into a merged box

CENTER_REUSE_PX = 20              # reuse a label only if center shift <= 10 px
MISS_TOLERANCE_FRAMES = 15         # keep a label alive up to 5 missed frames
HISTORY_FRAMES = 15                # label reuse memory window for names/indices

DRAW_SELECTED_COLOR = (0, 255, 0)  # GREEN for selected targets (bbox + label)
DRAW_UNSELECTED_COLOR = (0, 0, 255)  # RED for unselected targets (label only)
DRAW_TRAJ_COLOR = (255, 255, 0)   # YELLOW for trajectories
FONT = cv2.FONT_HERSHEY_SIMPLEX

def ensure_dir(p):
	os.makedirs(p, exist_ok=True)

# -------- utils --------
def iou_xyxy(a, b):
	x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
	x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
	iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
	inter = iw * ih
	ua = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
	ub = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
	denom = ua + ub - inter
	return inter / denom if denom > 0 else 0.0

def center_of(box):
	return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)

def dist2(p, q):
	dx = p[0] - q[0]; dy = p[1] - q[1]
	return dx*dx + dy*dy

def get_box_center(box):
	return center_of(box)

def get_box_area(box):
	return (box[2] - box[0]) * (box[3] - box[1])

def get_spatial_relationship(box1, box2):
	"""Get spatial relationship between two boxes"""
	c1 = get_box_center(box1)
	c2 = get_box_center(box2)
	
	dx = c2[0] - c1[0]
	dy = c2[1] - c1[1]
	
	# Calculate angle
	angle = math.atan2(dy, dx) * 180 / math.pi
	
	# Determine direction
	if -45 <= angle <= 45:
		return "right"
	elif 45 < angle <= 135:
		return "below"
	elif 135 < angle <= 180 or -180 <= angle <= -135:
		return "left"
	else:
		return "above"

def get_distance(box1, box2):
	c1 = get_box_center(box1)
	c2 = get_box_center(box2)
	return math.sqrt(dist2(c1, c2))

# -------- models --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.Compose([transforms.ToTensor()])

# YOLOv10 (fallback to v8 if weights are unavailable)
yolo = YOLO("yolov10n.pt")

frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(device)
retina = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT).eval().to(device)

# RF-DETR
from rfdetr import RFDETRBase
try:
	from rfdetr.util.coco_classes import COCO_CLASSES as RFDETR_COCO_CLASSES
except Exception:
	RFDETR_COCO_CLASSES = [f"class_{i}" for i in range(80)]

def class_id_to_name(class_id: int) -> str:
	try:
		if isinstance(RFDETR_COCO_CLASSES, (list, tuple)):
			if 0 <= class_id < len(RFDETR_COCO_CLASSES):
				return RFDETR_COCO_CLASSES[class_id]
		elif isinstance(RFDETR_COCO_CLASSES, dict):
			if class_id in RFDETR_COCO_CLASSES:
				return RFDETR_COCO_CLASSES[class_id]
			for name, cid in RFDETR_COCO_CLASSES.items():
				if cid == class_id:
					return name
	except Exception:
		pass
	return f"class_{class_id}"

try:
	rfdetr = RFDETRBase(device="cpu")
except TypeError:
	rfdetr = RFDETRBase()
	try:
		rfdetr.model.to("cpu")
	except Exception:
		pass
if hasattr(rfdetr, "optimize_for_inference"):
	try:
		rfdetr.optimize_for_inference()
	except Exception:
		pass

# BERT for description matching
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").eval().to(device)

@torch.no_grad()
def embed_text(text):
	tokens = tokenizer(text, return_tensors="pt").to(device)
	out = bert(**tokens).last_hidden_state.mean(dim=1)
	return out

# -------- detectors -> list of dict(box, score, class_id, source) --------
def detect_yolo(pil_img):
	arr = np.array(pil_img)
	results = yolo(arr)
	dets = []
	if len(results) > 0 and results[0].boxes is not None:
		b = results[0].boxes.xyxy.cpu().numpy()
		s = results[0].boxes.conf.cpu().numpy()
		c = results[0].boxes.cls.cpu().numpy().astype(int)  # 0..79
		for i in range(len(b)):
			dets.append({"box": b[i].tolist(), "score": float(s[i]), "class_id": int(c[i]), "source": "YOLO"})
	return dets

@torch.no_grad()
def detect_frcnn(pil_img):
	t = to_tensor(pil_img).to(device)
	out = frcnn([t])[0]
	b = out["boxes"].detach().cpu().numpy()
	s = out["scores"].detach().cpu().numpy()
	c = out["labels"].detach().cpu().numpy().astype(int)  # 1..91
	dets = []
	for i in range(len(b)):
		cid = int(c[i]) - 1  # map to ~0..90
		if 0 <= cid < 80:
			dets.append({"box": b[i].tolist(), "score": float(s[i]), "class_id": cid, "source": "FRCNN"})
	return dets

@torch.no_grad()
def detect_retina(pil_img):
	t = to_tensor(pil_img).to(device)
	out = retina([t])[0]
	b = out["boxes"].detach().cpu().numpy()
	s = out["scores"].detach().cpu().numpy()
	c = out["labels"].detach().cpu().numpy().astype(int)  # 1..91
	dets = []
	for i in range(len(b)):
		cid = int(c[i]) - 1
		if 0 <= cid < 80:
			dets.append({"box": b[i].tolist(), "score": float(s[i]), "class_id": cid, "source": "Retina"})
	return dets

@torch.no_grad()
def detect_rfdetr(pil_img, conf=0.5):
	d = rfdetr.predict(pil_img, threshold=conf)  # supervision.Detections
	dets = []
	if d is not None and len(d.xyxy) > 0:
		for i in range(len(d.xyxy)):
			dets.append({
				"box": list(map(float, d.xyxy[i])),
				"score": float(d.confidence[i]),
				"class_id": int(d.class_id[i]),  # 0..79
				"source": "RFDETR"
			})
	return dets

# -------- m-by-n consensus merge + clustering --------
SOURCE_PREF = ["RFDETR", "FRCNN", "YOLO", "Retina"]

def merge_detections_m_by_n(all_dets, m_sources=CONSENSUS_SOURCES_REQUIRED, consensus_iou=CONSENSUS_IOU, cluster_iou=CLUSTER_IOU, fallback_to_rfdetr=True, debug=False):
	if not all_dets:
		return []

	# Consensus gate: keep detections supported by >= m distinct sources (including self) at IoU >= consensus_iou
	kept_idx = []
	if m_sources <= 1:
		kept_idx = list(range(len(all_dets)))
	else:
		for i, di in enumerate(all_dets):
			sources = set()
			for j, dj in enumerate(all_dets):
				if i == j or iou_xyxy(di["box"], dj["box"]) >= consensus_iou:
					sources.add(dj["source"])
			if len(sources) >= m_sources:
				kept_idx.append(i)

	if debug:
		src_counts = Counter([d["source"] for d in all_dets])
		print(f"[merge] src counts: {dict(src_counts)} | kept after consensus: {len(kept_idx)} (m={m_sources})")

	# Fallback to RFDETR-only if requested and nothing passed consensus (common when m>1)
	if not kept_idx and fallback_to_rfdetr:
		rfd = [d for d in all_dets if d["source"] == "RFDETR"]
		if debug:
			print(f"[merge] fallback to RFDETR-only: {len(rfd)}")
		all_kept = rfd
	else:
		all_kept = [all_dets[i] for i in kept_idx] if kept_idx else []

	# If still empty, return nothing
	if not all_kept:
		return []

	# Cluster by IoU >= cluster_iou, average boxes, choose label by source preference within each cluster
	clusters = []
	used = [False] * len(all_kept)

	for i in range(len(all_kept)):
	    if used[i]:
		    continue
	    seed = all_kept[i]
	    group_idx = [i]
	    used[i] = True

	    # form group by IoU
	    for j in range(i + 1, len(all_kept)):
		    if used[j]:
			    continue
		    if iou_xyxy(seed["box"], all_kept[j]["box"]) >= cluster_iou:
			    group_idx.append(j)
			    used[j] = True

	    # average box and score
	    boxes = np.array([all_kept[k]["box"] for k in group_idx], dtype=float)
	    scores = np.array([all_kept[k]["score"] for k in group_idx], dtype=float)
	    avg_box = boxes.mean(axis=0).tolist()
	    avg_score = float(scores.mean())

	    # label by source priority
	    chosen_cid = None
	    for pref in SOURCE_PREF:
		    cands = [all_kept[k]["class_id"] for k in group_idx if all_kept[k]["source"] == pref]
		    if cands:
			    chosen_cid = Counter(cands).most_common(1)[0][0]
			    break
	    if chosen_cid is None:
		    chosen_cid = int(Counter([all_kept[k]["class_id"] for k in group_idx]).most_common(1)[0][0])

	    clusters.append({"box": avg_box, "score": avg_score, "class_id": chosen_cid})

	if debug:
		    print(f"[merge] clusters formed: {len(clusters)}")

	return clusters

# -------- Enhanced BERT-based command processing with MULTIPLE TARGET SUPPORT --------
class MultiTargetCommandProcessor:
	def __init__(self, tokenizer, bert_model, device):
		self.tokenizer = tokenizer
		self.bert_model = bert_model
		self.device = device
		
		# Pre-defined patterns for common commands
		self.patterns = {
			'multiple_targets': r'track\s+([a-z0-9_]+)\s+and\s+([a-z0-9_]+)',
			'spatial_relationship': r'track\s+target\s+(left|right|above|below|near|far)\s+of\s+([a-z0-9_]+)',
			'object_with_property': r'track\s+target\s+with\s+([a-z\s]+)',
			'switch_track': r'switch\s+track\s+to\s+target\s+(left|right|above|below|near|far)\s+of\s+([a-z0-9_]+)',
			'size_based': r'track\s+(largest|smallest|biggest|smallest)\s+target',
			'movement_based': r'track\s+target\s+(moving|stationary|fast|slow)',
			'color_based': r'track\s+(red|blue|green|yellow|white|black|orange|purple|pink|brown|gray|grey)\s+target',
			'class_based': r'track\s+(person|car|truck|bus|bike|motorcycle|ball|sports ball|basketball|soccer ball|tennis ball)',
			'specific_target': r'track\s+([a-z0-9_]+)',
		}
	
	@torch.no_grad()
	def embed_text(self, text):
		tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
		out = self.bert_model(**tokens).last_hidden_state.mean(dim=1)
		return out
	
	def parse_command(self, command, current_outputs, tracker):
		"""Parse and execute complex tracking commands - returns list of targets for multiple tracking"""
		command = command.lower().strip()
		
		# Extract current targets and their properties
		current_targets = {}
		for box, label, traj in current_outputs:
			# Get class_id from the tracker registry, with robust fallback
			class_id = 0  # default
			class_name = "person"  # default
			
			if label in tracker.registry:
				registry_entry = tracker.registry[label]
				
				# Try to get class_id from registry
				if 'class_id' in registry_entry:
					class_id = registry_entry['class_id']
				elif 'base' in registry_entry:
					# Parse class from base name
					base_name = registry_entry['base']
					class_id = self._get_class_id_from_name(base_name)
					class_name = base_name
				else:
					# Fallback: try to parse from label name
					class_id = self._get_class_id_from_label_name(label)
					class_name = self._get_class_name_from_label(label)
			else:
				# Fallback: parse from label name
				class_id = self._get_class_id_from_label_name(label)
				class_name = self._get_class_name_from_label(label)
			
			current_targets[label] = {
				'box': box,
				'trajectory': traj,
				'center': get_box_center(box),
				'area': get_box_area(box),
				'class_name': class_name,
				'class_id': class_id
			}
		
		# Pattern matching
		for pattern_name, pattern in self.patterns.items():
			match = re.search(pattern, command)
			if match:
				return self._execute_pattern(pattern_name, match, current_targets, command)
		
		# Fallback to simple BERT-based matching
		return self._bert_fallback(command, current_targets)
	
	def _get_class_id_from_label_name(self, label):
		"""Extract class_id from label name like 'person_1' -> 0"""
		# Remove numbers and underscores, get base class
		base_name = re.sub(r'_\d+$', '', label)
		return self._get_class_id_from_name(base_name)
	
	def _get_class_name_from_label(self, label):
		"""Extract class name from label like 'person_1' -> 'person'"""
		# Remove numbers and underscores, get base class
		base_name = re.sub(r'_\d+$', '', label)
		return base_name
	
	def _get_class_id_from_name(self, class_name):
		"""Get class_id from class name"""
		class_mapping = {
			'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
			'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
			'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
			'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19,
			'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24,
			'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29,
			'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
			'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38,
			'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43,
			'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
			'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53,
			'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58,
			'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
			'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68,
			'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73,
			'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78,
			'toothbrush': 79
		}
		return class_mapping.get(class_name.lower(), 0)  # default to person
	
	def _execute_pattern(self, pattern_name, match, current_targets, command):
		"""Execute specific pattern-based commands - returns list of targets for multiple tracking"""
		if pattern_name == 'multiple_targets':
			# track person_3 and person_4 -> return BOTH targets
			label1, label2 = match.groups()
			return self._find_multiple_targets([label1, label2], current_targets)
		
		elif pattern_name == 'spatial_relationship':
			# track target left/right/above/below of target_a
			direction, reference_label = match.groups()
			result = self._find_spatial_target(current_targets, reference_label, direction)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'object_with_property':
			# track target with ball
			property_desc = match.group(1)
			result = self._find_target_with_property(current_targets, property_desc)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'switch_track':
			# switch track to target left/right of target_a
			direction, reference_label = match.groups()
			result = self._find_spatial_target(current_targets, reference_label, direction)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'size_based':
			# track largest/smallest target
			size_type = match.group(1)
			result = self._find_size_based_target(current_targets, size_type)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'movement_based':
			# track target moving/stationary
			movement_type = match.group(1)
			result = self._find_movement_based_target(current_targets, movement_type)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'color_based':
			# track red/blue target
			color = match.group(1)
			result = self._find_color_based_target(current_targets, color)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'class_based':
			# track person/car/ball
			class_name = match.group(1)
			result = self._find_class_based_target(current_targets, class_name)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		elif pattern_name == 'specific_target':
			# track person_3
			target_label = match.group(1)
			result = self._find_specific_target(current_targets, target_label)
			return [result[0]] if result[0] else [], result[1] if result[0] else 0.0
		
		return [], 0.0
	
	def _find_multiple_targets(self, target_labels, current_targets):
		"""Find multiple specific targets - returns list of targets"""
		matches = []
		confidences = []
		
		for label in target_labels:
			if label in current_targets:
				matches.append(label)
				confidences.append(1.0)  # Perfect match
			else:
				# Try fuzzy matching
				best_match = None
				best_sim = -1.0
				for current_label in current_targets.keys():
					sim = self._calculate_similarity(label, current_label)
					if sim > best_sim:
						best_sim = sim
						best_match = current_label
				if best_match and best_sim > 0.5:  # Threshold for fuzzy matching
					matches.append(best_match)
					confidences.append(best_sim)
		
		# Return list of targets and average confidence
		avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
		return matches, avg_confidence
	
	def _find_specific_target(self, current_targets, target_label):
		"""Find a specific target by exact or fuzzy match"""
		if target_label in current_targets:
			return target_label, 1.0
		
		# Fuzzy matching
		best_match = None
		best_sim = -1.0
		for current_label in current_targets.keys():
			sim = self._calculate_similarity(target_label, current_label)
			if sim > best_sim:
				best_sim = sim
				best_match = current_label
		
		return best_match, best_sim if best_sim > 0.5 else 0.0
	
	def _find_spatial_target(self, current_targets, reference_label, direction):
		"""Find target based on spatial relationship"""
		if reference_label not in current_targets:
			return None, 0.0
		
		ref_box = current_targets[reference_label]['box']
		best_match = None
		best_score = -1.0
		
		for label, props in current_targets.items():
			if label == reference_label:
				continue
			
			relationship = get_spatial_relationship(ref_box, props['box'])
			if relationship == direction:
				# Calculate score based on distance and confidence
				distance = get_distance(ref_box, props['box'])
				score = 1.0 / (1.0 + distance / 100.0)  # Normalize distance
				if score > best_score:
					best_score = score
					best_match = label
		
		return best_match, best_score
	
	def _find_target_with_property(self, current_targets, property_desc):
		"""Find target with specific property using BERT"""
		best_match = None
		best_sim = -1.0
		
		prop_emb = self.embed_text(property_desc)
		
		for label, props in current_targets.items():
			class_emb = self.embed_text(props['class_name'])
			sim = torch.cosine_similarity(prop_emb, class_emb, dim=1).item()
			if sim > best_sim:
				best_sim = sim
				best_match = label
		
		return best_match, best_sim
	
	def _find_size_based_target(self, current_targets, size_type):
		"""Find largest or smallest target"""
		if not current_targets:
			return None, 0.0
		
		areas = [(label, props['area']) for label, props in current_targets.items()]
		areas.sort(key=lambda x: x[1], reverse=(size_type in ['largest', 'biggest']))
		
		return areas[0][0], 1.0
	
	def _find_movement_based_target(self, current_targets, movement_type):
		"""Find target based on movement characteristics"""
		best_match = None
		best_score = -1.0
		
		for label, props in current_targets.items():
			traj = props['trajectory']
			if len(traj) < 2:
				continue
			
			# Calculate movement metrics
			recent_points = traj[-5:] if len(traj) >= 5 else traj
			total_distance = 0.0
			for i in range(1, len(recent_points)):
				total_distance += math.sqrt(dist2(recent_points[i-1], recent_points[i]))
			
			avg_speed = total_distance / len(recent_points) if len(recent_points) > 1 else 0.0
			
			# Score based on movement type
			if movement_type == 'moving' and avg_speed > 5.0:
				score = min(avg_speed / 20.0, 1.0)
			elif movement_type == 'stationary' and avg_speed < 2.0:
				score = 1.0 - min(avg_speed / 5.0, 1.0)
			elif movement_type == 'fast' and avg_speed > 15.0:
				score = min(avg_speed / 30.0, 1.0)
			elif movement_type == 'slow' and avg_speed < 10.0:
				score = 1.0 - min(avg_speed / 15.0, 1.0)
			else:
				score = 0.0
			
			if score > best_score:
				best_score = score
				best_match = label
		
		return best_match, best_score
	
	def _find_color_based_target(self, current_targets, color):
		"""Find target based on color (simplified - would need actual color analysis)"""
		# This is a simplified implementation
		# In practice, you'd analyze the actual image regions for color
		best_match = None
		best_sim = -1.0
		
		color_emb = self.embed_text(color)
		
		for label, props in current_targets.items():
			class_emb = self.embed_text(props['class_name'])
			sim = torch.cosine_similarity(color_emb, class_emb, dim=1).item()
			if sim > best_sim:
				best_sim = sim
				best_match = label
		
		return best_match, best_sim
	
	def _find_class_based_target(self, current_targets, class_name):
		"""Find target based on class name"""
		best_match = None
		best_sim = -1.0
		
		class_emb = self.embed_text(class_name)
		
		for label, props in current_targets.items():
			label_emb = self.embed_text(props['class_name'])
			sim = torch.cosine_similarity(class_emb, label_emb, dim=1).item()
			if sim > best_sim:
				best_sim = sim
				best_match = label
		
		return best_match, best_sim
	
	def _bert_fallback(self, command, current_targets):
		"""Fallback to simple BERT-based matching"""
		best_match = None
		best_sim = -1.0
		
		cmd_emb = self.embed_text(command)
		
		for label, props in current_targets.items():
			label_emb = self.embed_text(label)
			sim = torch.cosine_similarity(cmd_emb, label_emb, dim=1).item()
			if sim > best_sim:
				best_sim = sim
				best_match = label
		
		return [best_match] if best_match else [], best_sim
	
	def _calculate_similarity(self, label1, label2):
		"""Calculate similarity between two labels"""
		emb1 = self.embed_text(label1)
		emb2 = self.embed_text(label2)
		return torch.cosine_similarity(emb1, emb2, dim=1).item()

# -------- identity + trajectories --------
class TrackRegistry:
    def __init__(self, center_reuse_px=CENTER_REUSE_PX, miss_tol=MISS_TOLERANCE_FRAMES, history_frames=HISTORY_FRAMES):
        self.center_reuse_px2 = center_reuse_px * center_reuse_px
        self.miss_tol = miss_tol
        self.history_frames = history_frames
        self.registry = {}  # label -> state dict {base, last_center, last_frame, missing, trajectory, last_velocity, class_id}
        self.next_idx = defaultdict(lambda: 1)

    def _base_name(self, class_id):
        return class_id_to_name(int(class_id))

    def _alloc_label(self, base):
        label = f"{base}_{self.next_idx[base]}"
        self.next_idx[base] += 1
        return label

    def step(self, frame_idx, merged_dets, selected_labels=None):
        matched = set()
        new_registry = {}

        # Candidates: labels seen within the miss tolerance
        candidates = []
        for label, st in self.registry.items():
            if st["missing"] <= self.miss_tol:
                candidates.append(label)

        # Prepare detections
        det_centers = [center_of(d["box"]) for d in merged_dets]
        det_bases = [self._base_name(d["class_id"]) for d in merged_dets]

        pairs = []

        for i, (cxy, base) in enumerate(zip(det_centers, det_bases)):

            for cand in candidates:
                st = self.registry[cand]
                if st["base"] != base:
                    continue

                d2 = dist2(cxy, st["last_center"])
                if d2 > self.center_reuse_px2:
                    continue

                # Calculate motion vectors for direction similarity
                traj = st["trajectory"]

                # Candidate velocity vector (last motion)
                if len(traj) >= 2:
                    vel1 = np.array(traj[-1]) - np.array(traj[-2])
                else:
                    vel1 = np.array([0.0, 0.0])

                # Detection velocity vector: difference between current detection and candidate last center
                vel2 = np.array(cxy) - np.array(st["last_center"])

                # Normalize helper
                def normalize(v):
                    norm = np.linalg.norm(v)
                    return v / norm if norm > 0 else v

                vel1_norm = normalize(vel1)
                vel2_norm = normalize(vel2)

                direction_similarity = np.dot(vel1_norm, vel2_norm)  # cosine similarity (-1 to 1)

                # Cost function: lower cost is better
                w = 1000  # weight for direction similarity (tune this)
                cost = d2 - w * direction_similarity

                pairs.append((cost, i, cand))

        pairs.sort(key=lambda x: x[0])  # sort by cost ascending

        det_assigned = set()
        cand_used = set()
        reuse_for_det = {}

        for cost, i, cand in pairs:
            if i in det_assigned or cand in cand_used:
                continue
            reuse_for_det[i] = cand
            det_assigned.add(i)
            cand_used.add(cand)

        outputs = []
        for i, det in enumerate(merged_dets):
            base = det_bases[i]
            box = det["box"]
            cxy = det_centers[i]
            class_id = det["class_id"]

            if i in reuse_for_det:
                lbl = reuse_for_det[i]
                st_prev = self.registry[lbl]
                traj = st_prev["trajectory"] + [cxy]

                if len(traj) >= 2:
                    last_velocity = np.array(traj[-1]) - np.array(traj[-2])
                else:
                    last_velocity = np.array([0.0, 0.0])

                new_registry[lbl] = {
                    "base": base,
                    "last_center": cxy,
                    "last_frame": frame_idx,
                    "missing": 0,
                    "trajectory": traj,
                    "last_velocity": last_velocity.tolist(),
                    "class_id": class_id,  # Store class_id in registry
                }
                matched.add(lbl)
                outputs.append((box, lbl, traj))
            else:
                # New track
                lbl = self._alloc_label(base)
                new_registry[lbl] = {
                    "base": base,
                    "last_center": cxy,
                    "last_frame": frame_idx,
                    "missing": 0,
                    "trajectory": [cxy],
                    "last_velocity": [0.0, 0.0],
                    "class_id": class_id,  # Store class_id in registry
                }
                matched.add(lbl)
                outputs.append((box, lbl, new_registry[lbl]["trajectory"]))

        # Propagate unmatched tracks with incremented missing count
        for lbl, st in self.registry.items():
            if lbl in matched:
                continue
            st2 = dict(st)
            st2["missing"] = st["missing"] + 1
            new_registry[lbl] = st2

        self.registry = new_registry

        # Check if any selected targets are lost
        selected_lost = False
        if selected_labels is not None:
            for selected_label in selected_labels:
                st = self.registry.get(selected_label)
                if st is None or st["missing"] > self.miss_tol:
                    selected_lost = True
                    break

        return outputs, selected_lost

# -------- main --------
def main():
	ensure_dir(OUTPUT_DIR)
	image_paths = sorted(
		[p for p in glob.glob(os.path.join(INPUT_DIR, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
	)
	if not image_paths:
		print(f"No images found in {INPUT_DIR}")
		return

	tracker = TrackRegistry()
	command_processor = MultiTargetCommandProcessor(tokenizer, bert, device)
	selected_labels = []  # List for multiple targets
	track_only_selected = False
	paused = False
	last_outputs = []

	cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

	frame_idx = 0
	while frame_idx < len(image_paths):
		
		if frame_idx >= len(image_paths):
			break
		img_path = image_paths[frame_idx]
		pil_img = Image.open(img_path).convert("RGB")
		frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

		if not paused:
			# 1) run 4 models
			yolo_d = detect_yolo(pil_img)
			frcnn_d = detect_frcnn(pil_img)
			retina_d = detect_retina(pil_img)
			rfdetr_d = detect_rfdetr(pil_img, conf=0.5)

			# 2) m-by-n consensus merge, then cluster
			all_dets = yolo_d + frcnn_d + retina_d + rfdetr_d
			merged = merge_detections_m_by_n(
				all_dets,
				m_sources=CONSENSUS_SOURCES_REQUIRED,
				consensus_iou=CONSENSUS_IOU,
				cluster_iou=CLUSTER_IOU,
				fallback_to_rfdetr=True,
				debug=False
			)

			# 3) update identities and trajectories with center-distance reuse and miss tolerance
			outputs, selected_lost = tracker.step(frame_idx, merged, selected_labels=selected_labels)

			# 4) MODIFIED: Always show ALL targets, but filter only for tracking logic
			# We don't filter outputs here - we show all targets in the drawing section

			# 5) draw with enhanced visualization
			canvas = frame_bgr.copy()
			
			# Draw all targets with different styles based on selection status
			for box, lbl, traj in outputs:
				b = list(map(int, box))
				
				# Determine if this target is selected
				is_selected = lbl in selected_labels
				
				if is_selected:
					# SELECTED TARGETS: Green bbox + label + trajectory
					color = DRAW_SELECTED_COLOR
					# Draw bounding box
					cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color, 2)
					# Draw label
					cv2.putText(canvas, lbl, (b[0], b[1]-8), FONT, 0.6, color, 2)
					# Draw trajectory
					traj_to_draw = traj[-15:] if len(traj) > 15 else traj
					for t in range(1, len(traj_to_draw)):
						p1 = (int(traj_to_draw[t-1][0]), int(traj_to_draw[t-1][1]))
						p2 = (int(traj_to_draw[t][0]), int(traj_to_draw[t][1]))
						cv2.line(canvas, p1, p2, DRAW_TRAJ_COLOR, 2)
				else:
					# UNSELECTED TARGETS: Only red label (no bbox, no trajectory)
					color = DRAW_UNSELECTED_COLOR
					# Draw only label at center of box
					center_x = (b[0] + b[2]) // 2
					center_y = (b[1] + b[3]) // 2
					cv2.putText(canvas, lbl, (center_x, center_y), FONT, 0.5, color, 1)

			cv2.imshow("Tracking", canvas)

			# 6) save
			out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
			Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

			frame_idx += 1
			last_outputs = outputs

		# key handling
		key = cv2.waitKeyEx(50) & 0xFF
		if key == ord('q'):
			break
		elif key == ord('p'):
			paused = not paused
		elif key == ord('c'):
			paused = True
			current_labels = [lbl for (_, lbl, _) in last_outputs] if last_outputs else list(tracker.registry.keys())
			print("\n[COMMAND] Available labels:", current_labels)
			print("Enhanced commands available:")
			print("  - track person_3 and person_4 (track multiple targets)")
			print("  - track target left/right/above/below of person_1")
			print("  - track target with ball")
			print("  - switch track to target left/right of person_1")
			print("  - track largest/smallest target")
			print("  - track target moving/stationary/fast/slow")
			print("  - track red/blue/green target")
			print("  - track person/car/ball")
			print("  - track person_3 (track specific target)")
			print("  - reset (resume all targets)")
			print("  - quit")
			print("\n[VISUAL] Selected targets: GREEN bbox+label+trajectory")
			print("[VISUAL] Other targets: RED label only (no bbox)")
			cmd = input(">> ").strip()

			if cmd.lower() == "quit":
				break
			elif cmd.lower() == "reset":
				track_only_selected = False
				selected_labels = []
				print("[INFO] Reset. All targets will show bboxes and labels.")
			else:
				if len(tracker.registry) == 0:
					print("[WARN] No targets to select.")
				else:
					# Use enhanced command processor - now returns list of targets
					target_matches, confidence = command_processor.parse_command(cmd, last_outputs, tracker)
					if target_matches:
						selected_labels = target_matches
						track_only_selected = True
						if len(target_matches) == 1:
							print(f"[INFO] Tracking '{target_matches[0]}' (confidence={confidence:.2f})")
							print(f"[VISUAL] '{target_matches[0]}' will show GREEN bbox+label+trajectory")
							print(f"[VISUAL] All other targets will show RED labels only")
						else:
							print(f"[INFO] Tracking multiple targets: {target_matches} (avg confidence={confidence:.2f})")
							print(f"[VISUAL] Selected targets will show GREEN bbox+label+trajectory")
							print(f"[VISUAL] All other targets will show RED labels only")
					else:
						print("[WARN] No match found for command.")
			paused = False

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()