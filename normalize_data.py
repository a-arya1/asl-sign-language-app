def normalize_landmarks(data):
    wristx, wristy, wristz = data[0], data[1], data[2]
    normalized_data = []
    for i in range(0, len(data), 3):
        normalized_data.append(data[i] - wristx)
        normalized_data.append(data[i+1] - wristy)
        normalized_data.append(data[i+2] - wristz)
    return normalized_data