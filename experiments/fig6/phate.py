import numpy as np
import matplotlib.pyplot as plt
import tphate
import pandas

def load_protein_features():
    df = pandas.read_csv("./features_dataframe.csv")
    print(df)

def main():
    load_protein_features()

if __name__=="__main__":
    main()