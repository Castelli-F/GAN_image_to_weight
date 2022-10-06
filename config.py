import inspect
import time
import os

# MAIN ROOT
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir = os.path.dirname(os.path.dirname(current_dir)) + "/dataset/"
print("la current directory è:", current_dir)
print("la dataset directory è:", dataset_dir)
# LOSS PLOT DIRECTORY
PLOT_PATH = current_dir + "/LOSS"
# IMG DIRECTORY
IMG_DIR = dataset_dir + "weight_strawberry/"
# MODEL SAVE DIRECTORY
ENCODED_PATH = current_dir + "/MODEL"
# RECONSTRUCTED IMAGES DIRECTORY
IMAGES = current_dir + "/RECONSTRUCTED IMAGES"
