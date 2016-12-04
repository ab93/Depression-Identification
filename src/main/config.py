import os

TRAIN_SPLIT_FILE = os.path.join('data','classification_data','training_split.csv')
TEST_SPLIT_FILE = os.path.join('data','classification_data','test_split.csv')
VAL_SPLIT_FILE = os.path.join('data','classification_data','dev_split.csv')
D_ND_DIR = os.path.join('data','disc_nondisc')
POS_NEG_DIR = os.path.join('data','pos_neg')
SEL_FEAT_TRAIN_REGULAR = os.path.join('data','selected_features','regular','train')
SEL_FEAT_VAL_REGULAR = os.path.join('data','selected_features','regular','val')
SEL_FEAT_TRAIN_NORMALIZED = os.path.join('data','selected_features','normalize','train')
SEL_FEAT_VAL_NORMALIZED = os.path.join('data','selected_features','normalize','val')
SEL_FEAT = os.path.join('data','selected_features')
ANOVA_DIR = os.path.join('results','anova')
GRID_SEARCH_DIR = os.path.join('results','grid_search')





# TRAIN_SPLIT_FILE = 'data/classification_data/training_split.csv'
# TEST_SPLIT_FILE = 'data/classification_data/test_split.csv'
# VAL_SPLIT_FILE = 'data/classification_data/dev_split.csv'
# D_ND_DIR = 'data/disc_nondisc'
# POS_NEG_DIR = 'data/pos_neg'