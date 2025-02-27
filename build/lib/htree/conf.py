# confg.py
# Default parameters
############################### DIRECTORIES ###############################
LOG_DIRECTORY = 'tmp/log'
OUTPUT_DIRECTORY = "tmp/Results"
OUTPUT_FIGURES_DIRECTORY = "tmp/Images"
OUTPUT_VIDEO_DIRECTORY = "tmp/Movies"
OUTPUT_SUBSPACE_DIRECTORY = "tmp/Subapce"
############################### TREE ###############################
ENABLE_ACCURATE_OPTIMIZATION = False
TOTAL_EPOCHS  = 100
INITIAL_LEARNING_RATE = 0.01
MAX_RANGE = 10
ENABLE_SAVE_MODE = False
ENABLE_VIDEO_EXPORT = False

CURV_RATIO = 0.5
NO_WEIGHT_RATIO = 0.5
EPSILON = 10**(-12)

DIMENSION = 2
WINDOW_RATIO = 0.025
INCREASE_FACTOR = 1.001
DECREASE_FACTOR = 0.98
INCREASE_COUNT_RATIO = 0.1

VIDEO_LENGTH = 60
############################### EMBEDDING ###############################
POINCARE_DOMAIN = (0,1)
LOID_DOMAIN = (-1-10**(-3), -1+10**(-3))
FRECHET_LEARNING_RATE = 0.001
FRECHET_MAX_EPOCHS = 1000
FRECHET_ERROR_TOLERANCE = 1e-8
ERROR_TOLERANCE = 1e-6

# Add other default parameters as needed






