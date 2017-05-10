import pandas as pd
from ..main import config
import os

def add_labels():
	split_file = config.TEST_SPLIT_FILE
	split_df = pd.read_csv(split_file)
	combined_file = os.path.join('data','classification_data','combined_WoZ+AI+Framing+PDHA_meta.csv')
	combined_df = pd.read_csv(combined_file)
	test_ids = split_df['participant_ID'].tolist()
	combined_df = combined_df[combined_df['Participant'].isin(test_ids)]
	combined_df = combined_df[['Participant','PTSD_binary','PTSD_score']]
	combined_df.rename(columns={'Participant':'Participant_ID' , 'PTSD_binary':'PHQ_Binary' , 'PTSD_score':'PHQ_Score'}, inplace=True)
	#print combined_df
	combined_df.to_csv(config.TEST_SPLIT_FILE, index=False)

if __name__ == '__main__':
	add_labels()