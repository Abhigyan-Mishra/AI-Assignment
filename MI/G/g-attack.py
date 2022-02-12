import argparse
import os
import pandas as pd

NUM_CLASSES = 1


def attack(model):
    print("Attacking")
    data = pd.read_csv("./wiscon.csv", index_col=False)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attack based on confidence values")
    parser.add_argument("-m", "--target_model", type=str,
                        default="../shadow_models/genetic/genetic_wiscon.h5", help="Path for shadow model")

    args = parser.parse_args()
    model = args.target_model

    if not model:
        print("PLEASE A TARGET MODEL/UNKNOWN MODEL")
        exit()

    attack(model)
    

