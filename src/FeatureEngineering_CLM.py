import csv
from io import StringIO
from numpy import *
import pandas as pd
import os
import sys
import openpyxl as px

valid_ques_bank = {}

ack_ques = set()
followup_ques = set()
intimate_ques = set()
non_intimate_ques = set()

question_time_bank = {}

features_CLM = []

class ques_time:

    def __init__(self, stime, etime):
        self.startTime = float(stime)
        self.endTime = float(etime)

    def getStartTime(self):
        return self.startTime

    def getEndTime(self):
        return self.endTime


def prepare_ques_dictionary(rootdir):

    global valid_ques_bank

    W = px.load_workbook(rootdir+'/Questions/QuestionsClassification.xlsx')
    p = W.get_sheet_by_name(name='Annotation-Supervised')

    a = []

    for row in p.iter_rows():
        if(row[0].internal_value == None):
            break

        valid_ques_bank[row[0].internal_value] = row[1].internal_value


def print_ques_bank():
    global valid_ques_bank

    for key in valid_ques_bank:
        #if valid_ques_bank[key] == 10:
        print(key, valid_ques_bank[key])



def parse_transcript(filepath):

    global valid_ques_bank, question_time_bank

    start_time = None
    stop_time = None
    ellie_ques = None

    prev_utterance = None
    utterance = None

    shud_continue = False

    s1 = 'so how are you doing today'

    with open(filepath, 'r') as csvfile:
        transcript = csv.reader(csvfile, delimiter='\t')

        for utter in transcript:
            #print(utterance)

            shud_continue = False

            prev_utterance = utterance
            utterance = utter

            if 'Ellie' in utterance:

                search_utter = utterance[3][utterance[3].find("(") + 1:utterance[3].find(")")]

                if search_utter in ack_ques:
                    shud_continue = True

                if search_utter in followup_ques:
                    shud_continue = True

                if shud_continue == True:
                    continue

                if start_time != None and ellie_ques != None:
                    stop_time = prev_utterance[1]
                    q = ques_time(start_time, stop_time)
                    question_time_bank[ellie_ques] = q

                    ellie_ques = None
                    start_time = None
                    stop_time = None

                #print('utterance3', utterance[3])

                for key in valid_ques_bank:

                    if key == search_utter:
                        start_time = utterance[0]
                        ellie_ques = key


        if start_time != None and ellie_ques != None:
            stop_time = prev_utterance[1]
            q = ques_time(start_time, stop_time)
            question_time_bank[ellie_ques] = q


            #prev_utterance = utterance


def parse_all_ques_types(rootdir):
    with open(rootdir+'/Questions/AllQues.csv', 'r') as csvfile:

        ellieUtterances = csv.reader(csvfile, delimiter=',')

        count = 0

        for utterance in ellieUtterances:

            if(count == 0 or count == 1):
                count = count+1
                continue

            if utterance[2] == '#follow_up':
                followup_ques.add(utterance[0])
            elif utterance[2] == '#ack':
                ack_ques.add(utterance[0])
            elif utterance[2] == '#non_int':
                non_intimate_ques.add(utterance[0])
            elif utterance[2] == '#int':
                intimate_ques.add(utterance[0])


def print_ques_sets():
    print('------------------------------------')
    print('follow up questions:\n')
    print(followup_ques)
    print('------------------------------------')

    print('------------------------------------')
    print('Acknowledgement Questions:\n')
    print(ack_ques)
    print('------------------------------------')

    print('------------------------------------')
    print('Non intimate questions:\n')
    print(non_intimate_ques)
    print('------------------------------------')

    print('------------------------------------')
    print('Initmate Questions:\n')
    print(intimate_ques)
    print('------------------------------------')



def print_ellie_ques_bank():

    for ques in question_time_bank:
        t = question_time_bank[ques]
        print(ques,' ', t.getStartTime(),' ', t.getEndTime())


def read_ques_time():
    f = open('U:/USC_Study/535/Project/Features/questions_timings.txt', "rb")
    bin_data = f.read()
    sio = StringIO(bin_data)
    graph_data = pickle.load(sio)
    print(graph_data)



def parse_CLM_features(dirpath, v_no):

    f_data = pd.read_csv(dirpath+'/'+v_no+'_CLM_features.txt')
    f3d_data = pd.read_csv(dirpath+'/'+v_no+'_CLM_features3D.txt')
    fgaze_data = pd.read_csv(dirpath+'/'+v_no+'_CLM_gaze.txt')
    fpose_data = pd.read_csv(dirpath+'/'+v_no+'_CLM_pose.txt')


    for ques in question_time_bank:

        if ques in valid_ques_bank:

            check = valid_ques_bank[ques]


            tt = question_time_bank[ques]

            t_start = tt.getStartTime()
            t_end = tt.getEndTime()


            # CLM Feature Extraction

            start_f = None
            end_f = None
            sum_f = None

            for i in range(len(f_data)):

                if f_data.iloc[i][1] >= float(t_start) and f_data.iloc[i][1] <= float(t_end):
                    if start_f == None:
                        start_f = i
                else:
                    if start_f != None:
                        end_f = i
                        break

            sum_f = f_data[start_f:end_f].mean()
            temp = list(sum_f)[1:]
            ft = [v_no, check, t_start, t_end]
            ft.extend(temp)

            sum_f3 = f3d_data[start_f:end_f].mean()
            temp = list(sum_f3)[1:]
            ft.extend(temp)

            sum_f_gaze = fgaze_data[start_f:end_f].mean()
            temp = list(sum_f_gaze)[1:]
            ft.extend(temp)

            sum_f_pose = fpose_data[start_f:end_f].mean()
            temp = list(sum_f_pose)[1:]
            ft.extend(temp)

            features_CLM.append(ft)

            print('\n', ques, '\n')



if __name__ == '__main__':

    rootdir = sys.argv[1]

    prepare_ques_dictionary(rootdir)
    print_ques_bank()

    parse_all_ques_types(rootdir)
    #print_ques_sets()

    #rootdir = 'U:/USC_Study/535/Project/Data'

    participant_id = None

    for subdir, dirs, files in os.walk(rootdir+"/Data"):

        participant_id = subdir[-5:-2]
        print(participant_id)

        t_path = participant_id+'_TRANSCRIPT.csv'

        if t_path in files:

            question_time_bank = {}

            transcript_path = os.path.join(subdir, t_path)
            parse_transcript(transcript_path)
            print(t_path)

            print_ellie_ques_bank()
            parse_CLM_features(subdir, participant_id)

        else:
            print(t_path, '  not found file')

    my_df = pd.DataFrame(features_CLM)
    my_df.to_csv(rootdir + '/' + "CLM_Feature_Extraction.csv", index=False, header=False)