
import json
import os
import numpy
import pandas
import shutil 
import argparse

from tqdm import tqdm

PDK_NAS = "~/mnt/pdk-nas"
OUTDIR = "../../../data"

def load_xlsx():
    df = pandas.read_excel("./xlsx/2023-02_Liste_neurones_MiniFinder.xlsx", sheet_name="liste_neurones_MiniFind_MLX")
    for key in ["Neuron-Id", "RawData"]:
        previous = df[key][0]
        for idx, row in df.iterrows():
            current = row[key]
            if isinstance(current, str) or not numpy.isnan(current):
                previous = current
            else:
                df.at[idx, key] = previous
    df["Neuron-Id"] = df["Neuron-Id"].astype(int)
    return df

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, help="Name of the condition")
    args = parser.parse_args()

    df = load_xlsx()

    condition = args.condition
    with open(f"./conditions/{condition}.json", "r") as file:
        conditions = json.load(file)
    for key, neuron_ids in conditions.items():

        # Creates output directory
        savedir = os.path.join(OUTDIR, "experiments", condition, key)
        os.makedirs(savedir, exist_ok=True)

        for neuron_id in tqdm(neuron_ids, desc="Neuron Ids"):
            if isinstance(neuron_id, (tuple, list)):
                neuron_id, stream_condition = neuron_id
            else:
                stream_condition = None
            subdf = df[df["Neuron-Id"] == neuron_id]

            if isinstance(stream_condition, str):
                subsubdf = subdf[subdf["Stream-Condition"] == stream_condition]
                if (len(subsubdf) > 1) or (len(subsubdf) < 1):
                    print("Neuron not found: ", neuron_id)
                else:
                    for idx, row in subsubdf.iterrows():
                        path = os.path.join(row["RawData"].replace("\\", "/"), row["Stream"])
                        path = path.replace("//PDK-NAS/Users", PDK_NAS)
                        path = "".join((path, ".tif"))
                        path = os.path.expanduser(path)
                    
                        # Copies the files from NAS
                        shutil.copy(path, os.path.join(savedir, f"{neuron_id}-{idx}.tif"))
            else:
                for idx, row in subdf.iterrows():
                    path = os.path.join(row["RawData"].replace("\\", "/"), row["Stream"])
                    path = path.replace("//PDK-NAS/Users", PDK_NAS)
                    path = "".join((path, ".tif"))
                    path = os.path.expanduser(path)

                    # Copies the files from NAS
                    shutil.copy(path, os.path.join(savedir, f"{neuron_id}-{idx}.tif"))

if __name__ == "__main__":

    main()
