# config.py
# Default parameters
############################### LOGGER ###############################
LOG_DIR = 'tmp/log'
############################### TREE ###############################
ACCURATE = False
TOTAL_EPOCHS  = 100
LEARNING_RATE_INIT = 0.01
MAX_DIAM = 10
ENABLE_SAVE = False
ENABLE_MOVIE = False
CURV_RATIO = 0.5
NO_WEIGHT_RATIO = 0.5
EPS = 10**(-12)

DIMENSION = 2
WINDOW_RATIO = 0.025
INCREASE_FACTOR = 1.001
DECREASE_FACTOR = 0.98
INCREASE_COUNT_RATIO = 0.1


RESULTS_DIR = "tmp/Results"
IMAGE_DIR = "tmp/Images"
MOVIE_DIR = "tmp/Movies"
SUBSPACE_DIR = "tmp/Subapce"

MOVIE_LENGTH = 60
############################### EMBEDDING ###############################

poincare_domain = (0,1)
loid_domain = (-1-10**(-3), -1+10**(-3))
frechet_lr = 0.001
frechet_max_iter = 1000
frechet_tol = 1e-8
norm_projection = 0.999999
atol = 1e-6



# Add other default parameters as needed






