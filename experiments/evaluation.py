import numpy as np
import matplotlib.pyplot as plt
from train import UNet3D
import csv
import os
import pandas


QC_model_folder = "../data/baselines/ZeroCostDL4Mic/unet3D-ZeroCostDL4Mic_complete_1-8_46"
QC_model_name = os.path.basename(QC_model_folder)
QC_model_path = os.path.dirname(QC_model_folder)

def load_model():
    full_QC_model_path = QC_model_path+"/"+QC_model_name+"/"
    if os.path.exists(full_QC_model_path):
        print("The "+QC_model_name+" network will be evaluated")
        model = UNet3D.from_path(QC_model_path, QC_model_name)
        model = model.to(device)
    else:
         W  = '\033[0m'  # white (normal)
        R  = '\033[31m' # red
        print(R+'!! WARNING: The chosen model does not exist !!'+W)
        print('Please make sure you provide a valid model path and model name before proceeding further.')
    return model


def inspect_loss_function():
    with open(QC_model_path+"/"+QC_model_name+"/Quality Control/train-stats.csv", "r") as csvfile:
        df = pandas.read_csv(csvfile, delimiter=",")
        stepsDataFromCSV = df["step"]
        lossDataFromCSV = df["loss"]
    
    with open(QC_model_path+'/'+QC_model_name+'/Quality Control/valid-stats.csv','r') as csvfile:
        df = pandas.read_csv(csvfile, delimiter=',')
        valstepsDataFromCSV = df["step"]
        vallossDataFromCSV = df["loss"]
        vallossstdDataFromCSV = df["std-loss"]
        
        fig = plt.figure(figsize=(15 ,10))
        plt.subplot(2, 1, 1)
        plt.plot(stepsDataFromCSV,lossDataFromCSV, label='Training loss')
        plt.plot(valstepsDataFromCSV,vallossDataFromCSV, label='Validation loss')
        plt.fill_between(
            valstepsDataFromCSV, vallossDataFromCSV - vallossstdDataFromCSV, vallossDataFromCSV + vallossstdDataFromCSV,
            color="tab:orange", alpha=0.3
        )
        plt.title('Training loss and validation loss vs. epoch number (linear scale)')
        plt.ylabel('Loss')
        plt.xlabel('Steps')
        plt.legend()

        plt.subplot(2,1,2)
        plt.semilogy(stepsDataFromCSV,lossDataFromCSV, label='Training loss')
        plt.semilogy(valstepsDataFromCSV,vallossDataFromCSV, label='Validation loss')
        plt.fill_between(
            valstepsDataFromCSV, vallossDataFromCSV - vallossstdDataFromCSV, vallossDataFromCSV + vallossstdDataFromCSV,
            color="tab:orange", alpha=0.3
        )
        plt.title('Training loss and validation loss vs. epoch number (log scale)')
        plt.ylabel('Loss')
        plt.xlabel('Steps')
        plt.legend()
        fig.savefig(QC_model_path+'/'+QC_model_name+'/Quality Control/lossCurvePlots.png')
        
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device) # TODO: move checks for model existence to inspect_loss_function and remove this call
    inspect_loss_function()
    

if __name__=="__main__":
    main()