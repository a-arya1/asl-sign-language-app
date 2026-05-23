import numpy as np

from train_model import main


def predict_sign(model, landmark_data):
    landmark_data = np.array(landmark_data).reshape(1, -1)
    proba = model.predict_proba(landmark_data)[0]
    confidence = max(proba)
    if confidence > 0.35:
        return model.classes_[np.argmax(proba)], confidence
    return "", 0.0


if __name__ == "__main__":
    main()
