import pandas as pd

def get_test_data():
    test_data = pd.read_csv('data/classification_data/dev_split.csv')
    #print test_data

    test = test_data['Participant_ID'].tolist()
    #print test
    #test.append(video)
    clm_d = pd.read_csv('data/disc_nondisc/discriminative_CLM.csv')
    covarep_d = pd.read_csv('data/disc_nondisc/discriminative_COVAREP.csv')
    liwc_d = pd.read_csv('data/disc_nondisc/discriminative_LIWC.csv')

    clm_nd = pd.read_csv('data/disc_nondisc/nondiscriminative_CLM.csv')
    covarep_nd = pd.read_csv('data/disc_nondisc/nondiscriminative_COVAREP.csv')
    liwc_nd = pd.read_csv('data/disc_nondisc/nondiscriminative_LIWC.csv')

    for key in test:
        if not ((clm_nd['video'] == key).any()  ):
            print "visual ",key
        if not ((covarep_nd['video'] == key).any() ):
            print "acoustic ", key
        #print key
        if not((liwc_nd['video'] == key).any()):
            print "liwc ", key

get_test_data()



