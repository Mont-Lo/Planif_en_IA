import cv2

def preprocess_observation(obs):
    """Convert an RGB image to grayscale, resize, and flatten."""
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # Convertir en niveaux de gris
    obs_resized = cv2.resize(obs_gray, (84, 84))  # Redimensionner à (84,84)
    obs_flattened = obs_resized.flatten()  # Convertir en 1D
    return obs_flattened