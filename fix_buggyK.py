import os
import sys
import yaml
import numpy as np
import pickle


def load_obj(name):
    """Load something with Pickle."""

    with open(name, "rb") as f:
        return pickle.load(f)


def fix_data_converted_files(config_path):

    # load config file
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    
    if config["KalmanMode"]:
        raise ValueError("This script is not compatible with Kalman mode. Modify it manually for Kalman mode.")

    pert_sfs_path = os.path.join(
        config["OutputPath"],
        config["RunName"],
        "archive_perturbation_sfs",
        "pert_sf_1.npz",
    )

    data_converted_path = os.path.join(
        config["OutputPath"], config["RunName"], "inversion", "data_converted"
    )

    # rename data converted directory
    old_path_updated = data_converted_path + "_old"
    os.rename(data_converted_path, old_path_updated)

    # create new data converted directory
    os.makedirs(data_converted_path, exist_ok=True)

    # list files in data_converted directory
    partial_K_files = np.sort(os.listdir(old_path_updated))

    # loop through files and apply scaling matrix
    for file in partial_K_files:

        dat = load_obj(os.path.join(old_path_updated, file))

        pert_sfs = np.load(pert_sfs_path)["effective_pert_sf"]
        scaling_matrix = pert_sfs / (pert_sfs - 1)

        if config["OptimizeBCs"]:
            # add four elements for BCs
            scaling_matrix = np.append(scaling_matrix, np.ones(4))
            
        # Note: assumes hemispheric OH optimization, 
        # modify if using only single OH element
        if config["OptimizeOH"]:
            oh_pert = float(config["PerturbValueOH"])
            correct_oh_pert = oh_pert - 1
            # add two elements for OH
            scaling_matrix = np.append(
                scaling_matrix, (oh_pert / correct_oh_pert) * np.ones(2)
            )

        dat["K"] *= scaling_matrix

        # save updated file
        with open(os.path.join(data_converted_path, file), "wb") as f:
            pickle.dump(dat, f, pickle.HIGHEST_PROTOCOL)

        print (f"File {file} updated successfully.")


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
    except:
        raise ValueError(
            "Please provide a configuration file path. eg. python fix_buggyK.py config.yml"
        )

    # fix the bug
    fix_data_converted_files(config_path)