import pandas as pd
import numpy as np
import json
import os, sys

path = sys.argv[1]
o    = sys.argv[2]
print(f"GOT PATH: { path }")
print(f"will write to: { o }")

def reshape(data) -> dict:
    """ Gets dict of shape 
    {
        "began_logging_at"     : int,
        "logged_every"         : int,
        "weight_vectors"       : np.array,
        "optima_positions"     : np.array,
        "contribution_matrices": np.array,
    }
    """
    return {
        "weight_vectors"       : np.array(data['weight_vectors']).reshape(50, 1000, 4),
        "contribution_matrices": np.array(data['contribution_matrices']).reshape(50, 1000, 4, 4),
        "optima_positions"     : np.array(data['optima_positions'])
}

N = np.array
with open(path) as infile:
    data = reshape(json.load(infile))
all_lists = []

for i in range(50):

    l_index = []
    l_index.append(N( data['optima_positions'][i] ).tolist())
    # print("After optima pos:", len(l_index))

    for j in data['weight_vectors'][i]:
        l_index.append(np.array(j).tolist())
    # print("After weights:", len(l_index))

    for k in data['contribution_matrices'][i]:
        l_index.append(N(k).tolist())
    # print("After contribs :", len(l_index))
        
    all_lists.append(l_index)

cols = ["optima_at_t", 
        *["ind_{}_weight".format(x+1) for x in range(1000)],
        *["ind_{}_cont".format(x+1) for x in range(1000)]
        ]

df   = pd.DataFrame(all_lists, columns=cols)
df.to_csv(o)

print("saved {} successfully".format(o))
