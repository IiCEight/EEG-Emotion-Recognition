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



SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4','F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

HSLT_SEED_Regions = {
    'PF': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],
    'F':  ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    'LT': ['FT7', 'FC5', 'FC3', 'T7', 'C5', 'C1'], # C3
    'RT': ['FT8', 'FC4', 'FC6', 'T8', 'C2', 'C6', 'CP6'], # C4
    'C':  ['FC1', 'C3', 'CZ', 'FCZ', 'FC2', 'C4'],
    'LP': ['TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3', 'P1', 'PO3'],
    'P':  ['CP1', 'CP2', 'CPZ', 'PZ'],
    'RP': ['TP8', 'CP4', 'P8', 'P6', 'P2', 'P4', 'PO4'],
    'O':  ['PO7', 'PO5', 'POZ', 'PO6', 'PO8', 'CB1', 'O1', 'O2', 'OZ', 'CB2']
}


SEED_ADJACENCY_CHANNEL = {
    'FP1': ['FPZ', 'AF3'],
    'FPZ': ['FP1', 'FP2'],
    'FP2': ['FPZ', 'AF4'],
    'AF3': ['FP1', 'F5', 'F3', 'F1'],
    'AF4': ['F2', 'F4', 'F6', 'FP2'],
    'F7': ['F5', 'FT7'],
    'F5': ['F7', 'AF3', 'F3', 'FC5'],
    'F3': ['AF3', 'F5', 'FC3', 'F1'],
    'F1': ['AF3', 'F3', 'FC1', 'FZ'],
    'FZ': ['F1', 'FCZ', 'F2'],
    'F2': ['FZ', 'FC2', 'F4', 'AF4'],
    'F4': ['F2', 'FC4', 'F6', 'AF4'],
    'F6': ['AF4', 'F4', 'FC6', 'F8'],
    'F8': ['F6', 'FT8'],
    'FT7': ['F7', 'FC5', 'T7'],
    'FC5': ['F5', 'FT7', 'C5', 'FC3'],
    'FC3': ['F3', 'FC5', 'C3', 'FC1'],
    'FC1': ['F1', 'FC3', 'C1', 'FCZ'],
    'FCZ': ['FZ', 'FC1', 'CZ', 'FC2'],
    'FC2': ['F2', 'FCZ', 'C2', 'FC4'],
    'FC4': ['F4', 'FC2', 'C4', 'FC6'],
    'FC6': ['F6', 'FC4', 'C6', 'FT8'],
    'FT8': ['F8', 'FC6', 'T8'],
    'T7': ['FT7', 'C5', 'TP7'],
    'C5': ['FC5', 'T7', 'C3', 'CP5'],
    'C3': ['FC3', 'C5', 'C1', 'CP3'],
    'C1': ['FC1', 'C3', 'CP1', 'CZ'],
    'CZ': ['FCZ', 'C1', 'CPZ', 'C2'],
    'C2': ['FC2', 'CZ', 'CP2', 'C4'],
    'C4': ['FC4', 'C2', 'CP4', 'C6'],
    'C6': ['FC6', 'C4', 'CP6', 'T8'],
    'T8': ['FT8', 'C6', 'TP8'],
    'TP7': ['T7', 'CP5', 'P7'],
    'CP5': ['C5', 'TP7', 'P5', 'CP3'],
    'CP3': ['C3', 'CP5', 'P3', 'CP1'],
    'CP1': ['C1', 'CP3', 'P1', 'CPZ'],
    'CPZ': ['CZ', 'CP1', 'PZ', 'CP2'],
    'CP2': ['C2', 'CPZ', 'P2', 'CP4'],
    'CP4': ['C4', 'CP2', 'P4', 'CP6'],
    'CP6': ['C6', 'CP4', 'P6', 'TP8'],
    'TP8': ['T8', 'CP6', 'P8'],
    'P7': ['TP7', 'P5', 'PO7'],
    'P5': ['CP5', 'P7', 'PO5', 'P3'],
    'P3': ['CP3', 'P5', 'P1'],
    'P1': ['CP1', 'P3', 'PO3', 'PZ'],
    'PZ': ['CPZ', 'P1', 'POZ', 'P2'],
    'P2': ['CP2', 'PZ', 'PO4', 'P4'],
    'P4': ['CP4', 'P2', 'P6'],
    'P6': ['CP6', 'P4', 'P8'],
    'P8': ['TP8', 'P6', 'PO8'],
    'PO7': ['P7', 'PO5', 'CB1'],
    'PO5': ['P5', 'PO7', 'CB1', 'PO3'],
    'PO3': ['P1', 'PO5', 'O1', 'POZ'],
    'POZ': ['PZ', 'PO3', 'OZ', 'PO4'],
    'PO4': ['P2', 'POZ', 'O2', 'PO6'],
    'PO6': ['P6', 'PO4', 'CB2', 'PO8'],
    'PO8': ['P8', 'PO6', 'CB2'],
    'CB1': ['PO7', 'PO5', 'O1'],
    'O1': ['CB1', 'PO3', 'OZ'],
    'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['PO4', 'OZ', 'CB2'],
    'CB2': ['PO6', 'O2', 'PO8']
}

SEED_GLOBAL_CHANNEL_PAIRS = [
    ['FP1', 'FP2'],
    ['AF3', 'AF4'],
    ['F5', 'F6'],
    ['FC5', 'FC6'],
    ['C5', 'C6'],
    ['CP5', 'CP6'],
    ['P5', 'P6'],
    ['PO5', 'PO6'],
    ['O1', 'O2']
]

SEED_ADJACENCY_MATRIX = generate_adjacency_matrix(SEED_CHANNEL_NAME, SEED_ADJACENCY_CHANNEL)

SEED_RGNN_ADJACENCY_MATRIX = generate_rgnn_adjacency_matrix(channel_names=SEED_CHANNEL_NAME, channel_loc=system_10_05_loc, global_channel_pair=SEED_GLOBAL_CHANNEL_PAIRS)

