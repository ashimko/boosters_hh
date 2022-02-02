
#COLUMN NAMES
import os


TARGET = 'target'
CITY = 'city'
ID = 'review_id'
POSITION = 'position'
POSITION_AS_TXT = POSITION + '_txt'
POSITIVE = 'positive'
NEGATIVE = 'negative'
SALARY_RATING = 'salary_rating'
TEAM_RATING = 'team_rating'
MANAGMENT_RATING = 'managment_rating'
CAREER_RATING = 'career_rating'
WORKPLACE_RATING = 'workplace_rating'
REST_RECOVERY_RATING = 'rest_recovery_rating'
N_LABELS = 9
N_SPLITS = 3
RANDOM_STATE = 77

# COLUMN SETS
UNORDERED_CATEGORIES = [CITY, POSITION]
ORDERED_CATEGORIES = [SALARY_RATING, TEAM_RATING, MANAGMENT_RATING, CAREER_RATING,
                      WORKPLACE_RATING, REST_RECOVERY_RATING]
TEXT_COLS = [POSITIVE, NEGATIVE]

# PATH
ABS_PREFIX = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ABS_PREFIX, 'data')
ORIGINAL_DATA_PATH = os.path.join(DATA_PATH, 'original')
PREPARED_DATA_PATH = os.path.join(DATA_PATH, 'prepared')
MORPH_DATA_PATH = os.path.join(DATA_PATH, 'morph')
HANDCRAFTED_DATA_PATH = os.path.join(DATA_PATH, 'handcrafted')
MODEL_PATH = os.path.join(ABS_PREFIX, 'model_checkopoints')
SCORES_PATH = os.path.join(ABS_PREFIX, 'scores')
PLOTS_PATH = os.path.join(ABS_PREFIX, 'plots')
TEST_PRED_PATH = os.path.join(ABS_PREFIX, 'test_predictions')
SUBMITIONS_PATH = os.path.join(ABS_PREFIX, 'submitions')
OOF_PRED_PATH = os.path.join(ABS_PREFIX, 'oof_predictions')
METRICS = ('f1_samples', 'precision_samples', 'recall_samples', 'roc_auc_ovo', 'neg_log_loss')
