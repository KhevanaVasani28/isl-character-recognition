# %% =======
# imports 
# ==========
import cv2
import numpy as np
import mediapipe as mp
from config import *

# MediaPipe hand detector used by feature extraction
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5,
)

# %% ==========================================================
# STEP 1 : Function to Process Hands with Proper Ordering
# ==========================================================
def process_hands_correctly(multi_hand_landmarks, multi_handedness):
    """
    Process hands with proper left/right ordering.
    Returns: (left_hand_landmarks, right_hand_landmarks)
    """
    left_hand = None
    right_hand = None
    
    if multi_hand_landmarks is None:
        return left_hand, right_hand
    
    # Identify which hand is which
    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        handedness = multi_handedness[idx].classification[0].label
        if handedness == 'Left':
            left_hand = hand_landmarks
        else:  # 'Right'
            right_hand = hand_landmarks
    
    return left_hand, right_hand

# %% ==========================================================
# STEP 2 : Feature Extraction for Each Hand
# ==========================================================

def extract_hand_features(hand_landmarks, hand_type):
    """
    Extract features for a single hand.
    hand_type: 0 for left, 1 for right
    Returns: list of features [hand_type, x1, y1, z1, x2, y2, z2, ...]
    """
    if hand_landmarks is None:
        # Return zeros for all landmarks, but keep hand_type to indicate which hand it is
        features = [hand_type]  # 0 for left, 1 for right
        features.extend([0.0] * 63)  # 21 landmarks * 3 coordinates
        return features
    
    features = [hand_type]  # 0 for left, 1 for right
    
    # Wrist is landmark 0
    wrist = hand_landmarks.landmark[0]
    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z
    
    # Normalize all landmarks relative to wrist
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist_x,
            landmark.y - wrist_y,
            landmark.z - wrist_z
        ])
    
    return features

# %% =======================================================
# STEP 3 : Hand ordering 
# ==========================================================
# from static_utils

def extract_features_from_rgb(frame_rgb):
    """Returns 128-dim vector for one frame."""
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks is None:
        # No hand: left=type 0 + 63 zeros, right=type 1 + 63 zeros
        return np.array([0.0] + [0.0]*63 + [1.0] + [0.0]*63, dtype=np.float32)

    left, right = process_hands_correctly(
        results.multi_hand_landmarks,
        results.multi_handedness
    )
    left_feat = extract_hand_features(left, hand_type=0)
    right_feat = extract_hand_features(right, hand_type=1)
    return np.array(left_feat + right_feat, dtype=np.float32)

