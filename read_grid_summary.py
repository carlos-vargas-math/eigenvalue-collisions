import numpy as np

grid_search_summary_array = np.load("N=10seedFrom1000To1000&Curve.CIRCLE&Distribution=complexGaussian&Traceless=True.npy", allow_pickle=True)
# grid_search_summary_array = np.load('grid_search_summary/grid_search_summary.npy', allow_pickle=True)
# grid_search_summary_array = np.load('grid_search_summary/grid_search_summaryN5Circuit.npy', allow_pickle=True)

for row in grid_search_summary_array:  
    print(row[6])
