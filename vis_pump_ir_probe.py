import os
import numpy as np

def load_data_set(path):
    # Get delay count
    delay_path = os.path.join(path, "scans")
    n_delays = len(os.listdir(delay_path))
    print(n_delays)



if __name__ == "__main__":
    load_data_set("/Users/arthun/Downloads/20200822__009")