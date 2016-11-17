import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_combined_data(file1, file2):
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    print df1
    print df2

def main():
    get_combined_data(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
