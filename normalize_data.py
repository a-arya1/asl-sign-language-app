import numpy as np


def normalize_landmarks(data):
    wristx, wristy, wristz = data[0], data[1], data[2]
    normalized_data = []
    for i in range(0, len(data), 3):
        normalized_data.append(data[i] - wristx)
        normalized_data.append(data[i+1] - wristy)
        normalized_data.append(data[i+2] - wristz)

    ref_dist = np.sqrt(normalized_data[27]**2 + normalized_data[28]**2 + normalized_data[29]**2)
    if ref_dist > 0:
        normalized_data = [v / ref_dist for v in normalized_data]
    return normalized_data

def get_angle_features(lm):
    # lm = list of 21 (x,y) tuples
    def calc_angle(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
        return angle

    joints = [
        (2,3,4),
        (5,6,7), (6,7,8),
        (9,10,11), (10,11,12),
        (13,14,15), (14,15,16),
        (17,18,19), (18,19,20)
    ]
    angleList = []
    for trip in joints:
        a, b, c = trip
        angleList.append(calc_angle(lm[a], lm[b], lm[c]))
    return angleList
