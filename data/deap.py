import numpy as np
from .channel_location import system_10_05_loc
def generate_adjacency_matrix(channel_names, channel_adjacent):
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))
    for key, value in channel_adjacent.items():
        idx1 = np.where(channel_names == key)[0][0]
        for chan in value:
            idx2 = np.where(channel_names == chan)[0][0]
            adjacency_matrix[idx1][idx2] = 1
    return adjacency_matrix

def generate_rgnn_adjacency_matrix(channel_names, channel_loc, global_channel_pair):
    channel_names = np.array(channel_names)
    channel_num = len(channel_names)
    adjacency_matrix = np.zeros((channel_num, channel_num))
    for chan1 in channel_names:
        idx1 = np.where(channel_names == chan1)[0][0]
        for chan2 in channel_names:
            idx2 = np.where(channel_names == chan2)[0][0]
            if chan1 == chan2:
                adjacency_matrix[idx1][idx2] = 1
            else:
                cor1 = np.array(channel_loc[chan1])/10
                cor2 = np.array(channel_loc[chan2])/10
                dis_sq = 0
                for i in range(3):
                    dis_sq += np.square(cor1[i] - cor2[i])
                adjacency_matrix[idx1][idx2] = min(5/dis_sq, 1)
                adjacency_matrix[idx2][idx1] = min(5/dis_sq, 1)
    # print((np.where(adjacency_matrix > 0.1)[0].shape[0])/62/62)
    adjacency_matrix = differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair)
    return adjacency_matrix
def differential_asymmetry_leverage(channel_names, adjacency_matrix, global_channel_pair):
    for pair in global_channel_pair:
        idx1 = np.where(channel_names == pair[0])[0][0]
        idx2 = np.where(channel_names == pair[1])[0][0]
        adjacency_matrix[idx1][idx2] -= 1
        adjacency_matrix[idx2][idx1] -= 1
    return adjacency_matrix

DEAP_CHANNEL_NAME = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']

HSLT_DEAP_Regions = {
    'PF': ['FP1', 'AF3', 'AF4', 'FP2'],
    'F': ['F7', 'F3', 'FZ', 'F4', 'F8'],
    'LT': ['FC5', 'T7', 'CP5'],
    'C': ['FC1', 'C3', 'CZ', 'C4', 'FC2'],
    'RT': ['FC6', 'T8', 'CP6'],
    'LP': ['P7', 'P3', 'PO3'],
    'P': ['CP1', 'PZ', 'CP2'],
    'RP': ['P8', 'P4', 'PO4'],
    'O': ['O1', 'OZ', 'O2']
}
DEAP_GLOBAL_CHANNEL_PAIRS = [
    ['FP1', 'FP2'],
    ['AF3', 'AF4'],
    ['FC5', 'FC6'],
    ['CP5', 'CP6'],
    ['O1', 'O2']
]

DEAP_RGNN_ADJACENCY_MATRIX = generate_rgnn_adjacency_matrix(channel_names=DEAP_CHANNEL_NAME, channel_loc=system_10_05_loc,global_channel_pair=DEAP_GLOBAL_CHANNEL_PAIRS)




