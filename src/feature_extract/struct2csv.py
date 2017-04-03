import scipy.io as sio
import csv
import os
import math

sessions_path=os.path.join('..','..','..','Data_new')
sessions=os.listdir(sessions_path)
#header=['F0','VUV','NAQ','QOQ','H1H2','PSP','MDQ','peakSlope','Rd','Rd_conf','creak','MCEP_0','MCEP_1','MCEP_2','MCEP_3','MCEP_4','MCEP_5','MCEP_6','MCEP_7','MCEP_8','MCEP_9','MCEP_10','MCEP_11','MCEP_12','MCEP_13','MCEP_14','MCEP_15','MCEP_16','MCEP_17','MCEP_18','MCEP_19','MCEP_20','MCEP_21','MCEP_22','MCEP_23','MCEP_24','HMPDM_0','HMPDM_1','HMPDM_2','HMPDM_3','HMPDM_4','HMPDM_5','HMPDM_6','HMPDM_7','HMPDM_8','HMPDM_9','HMPDM_10','HMPDM_11','HMPDM_12','HMPDM_13','HMPDM_14','HMPDM_15','HMPDM_16','HMPDM_17','HMPDM_18','HMPDM_19','HMPDM_20','HMPDM_21','HMPDM_22','HMPDM_23','HMPDM_24','HMPDD_0','HMPDD_1','HMPDD_2','HMPDD_3','HMPDD_4','HMPDD_5','HMPDD_6','HMPDD_7','HMPDD_8','HMPDD_9','HMPDD_10','HMPDD_11','HMPDD_12']
header=['F1','F2','F3','F4','F5']
sessions=sessions[1:]
for session in sessions:
    if int(session[0:3]) >= 600:
        print session
        covarep_file=os.path.join(sessions_path,session,session[0:3]+'_COVAREP.mat')
        mat_contents=sio.loadmat(covarep_file)
        data=[]
        for h in header:
            current_feature=mat_contents['covarep'][h][0][0]
            zipped_feature=zip(*current_feature)
            final_feature=list(zipped_feature[0])
            for i in range(0,len(final_feature)):
                if math.isnan(final_feature[i]):
                    final_feature[i]=0
            data.append(final_feature)

        data=[list(a) for a in zip(*data)]
        print len(data)
        print len(data[0])
        print data[0][0]
        csv_path=os.path.join(sessions_path,session,session[0:3]+'_FORMANT.csv')
        with open(csv_path,'w') as f:
            writer=csv.writer(f)
            for row in data:
                writer.writerow(row)





