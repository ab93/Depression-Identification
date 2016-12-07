import os

TRAIN_SPLIT_FILE = os.path.join('data','classification_data','training_split.csv')
TEST_SPLIT_FILE = os.path.join('data','classification_data','test_split.csv')
VAL_SPLIT_FILE = os.path.join('data','classification_data','dev_split.csv')
D_ND_DIR = os.path.join('data','disc_nondisc')
POS_NEG_DIR = os.path.join('data','pos_neg')

SEL_FEAT_TRAIN_REGULAR_CLASSIFY = os.path.join('data','selected_features','regular','classify','train')
SEL_FEAT_TEST_REGULAR_CLASSIFY = os.path.join('data','selected_features','regular','classify','test')
SEL_FEAT_VAL_REGULAR_CLASSIFY = os.path.join('data','selected_features','regular','classify','val')
SEL_FEAT_TRAIN_REGULAR_ESTIMATE = os.path.join('data','selected_features','regular','estimate','train')
SEL_FEAT_VAL_REGULAR_ESTIMATE = os.path.join('data','selected_features','regular','estimate','val')
SEL_FEAT_TEST_REGULAR_ESTIMATE = os.path.join('data','selected_features','regular','estimate','test')


# SEL_FEAT_TRAIN_NORMALIZED_CLASSIFY = os.path.join('data','selected_features','normalize','classify','train')
# SEL_FEAT_VAL_NORMALIZED_CLASSIFY = os.path.join('data','selected_features','normalize','classify','val')
# SEL_FEAT_TRAIN_NORMALIZED_ESTIMATE = os.path.join('data','selected_features','normalize','estimate','train')
# SEL_FEAT_VAL_NORMALIZED_ESTIMATE = os.path.join('data','selected_features','normalize','estimate','val')

RESULTS_CLASSIFY = os.path.join('results','grid_search','classification')
RESULTS_ESTIMATE = os.path.join('results','grid_search','regression')


SEL_FEAT = os.path.join('data','selected_features')
ANOVA_DIR = os.path.join('results','anova')
GRID_SEARCH_CLF_DIR = os.path.join('results','grid_search','classification')
GRID_SEARCH_REG_DIR = os.path.join('results','grid_search','regression')
