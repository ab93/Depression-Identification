import os
import fnmatch
import sys

def renameCSVToTxt(dir):

    rootdir = dir

    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, '*CLM*.csv'):
            #print(os.path.join(rootdir+'/'+filename))
            os.rename(os.path.join(root+'/'+filename), root+'/'+filename[:-4] + '.txt')



if __name__ == "__main__":

    dirPath = sys.argv[1]
    renameCSVToTxt(dirPath)
