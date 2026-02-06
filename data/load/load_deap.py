from loguru import logger
import numpy as np

# for loading .dat files
#  pickle is a way to convert a complex Python object
# (like a list, a dictionary, or a massive 3D array of EEG data)
# into a byte stream.

# It "freezes" the data exactly as it exists in Python’s memory.
# When you load a DEAP .dat file, you don't just get text;
# you get a NumPy array and a Dictionary that are ready to be used immediately
# in your code without any parsing.
import pickle


def load_deap(dataset_path: str):
    """
    input file: 32 files contains 32 subject's eeg data

    output shape : (session(1), subject, trail, electrode, raw_data),
                   (session(1), subject, trail, label)

    return: data, label, sampling_rate, num_electrodes
    
    under dataset_path dir, it has 32.dat file, each represent one subject
    """
    logger.info(f"Loading DEAP dataset from path {dataset_path} ...")

    electrode_names = [
        "Fp1",
        "AF3",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "PO3",
        "O1",
        "Oz",
        "Pz",
        "Fp2",
        "AF4",
        "Fz",
        "F4",
        "F8",
        "FC6",
        "FC2",
        "Cz",
        "C4",
        "T8",
        "CP6",
        "CP2",
        "P4",
        "P8",
        "PO4",
        "O2",
    ]

    data = [[]]
    label = [[]]
    # Sampling Rate (128Hz)
    fs = 128
    # DEAP trials include a 3-second rest time before the stimulus
    pre_time = 3

    # 63 times 128Hz = 8064 data points per trial and electrode
    end_time = 63 
    pretrail = pre_time * fs

    # str(i) transforms integer i to a string
    # zfill(2) is a string method that adds leading zeros until
    # the string reaches a length of 2.
    # file name is like s01.dat.
    eeg_files = [f"s{str(i).zfill(2)}.dat" for i in range(1, 33)]
    for s_i, subject_file in enumerate(eeg_files):
        with open(f"{dataset_path}/" + subject_file, "rb") as f:
            sub_data = pickle.load(f, encoding="latin")

        # sub_data is a dictionary with keys 'data' and 'labels'
        # 'data' shape -> (trails(40), electrodes(32eeg, 8others, 40sum), raw_data(8064))
        # 'labels' shape -> (trails(40), label(4)(valence, arousal, dominance, liking))

        logger.debug(
            "Subject {} Data keys: {}, data type {}, data shape {}, label shape {}",
            s_i,
            sub_data.keys(),
            type(sub_data),
            np.array(sub_data["data"]).shape,
            np.array(sub_data["labels"]).shape,
        )

        # The Resting State: DEAP trials include a 3-second pre-stimulus
        #  period where the subject is resting.

        # Mean Calculation: This line takes the first 3 seconds (range(3))
        # of the 32 EEG electrodes and calculates their average signal.
        # This "baseline" represents the subject's brain activity before
        # the music video started i.e., the resting state.
        baseline = np.mean(
            [sub_data["data"][:, :32, i * fs : (i + 1) * fs] for i in range(3)], axis=0
        )

        # The Logic: It iterates through the entire signal and subtracts
        # the baseline from every second of the recording.

        # Purpose: This removes the unique "resting DC offset" of each
        # subject. It ensures the neural network focuses on the change
        # in brain activity caused by the emotion, rather than the
        # subject's permanent brain wave characteristics
        for sec in range(pre_time, end_time):
            sub_data["data"][:, :32, sec * fs : (sec + 1) * fs] -= baseline

        sub_data_list = []
        sub_label_list = []

        # Use zip to combine data and labels for each trial into a tuple
        for t_i, (trail_data, trail_label) in enumerate(
            zip(sub_data["data"], sub_data["labels"])
        ):
            # trail_data shape->(electrodes(32eeg, 8peripheral), raw_data)
            # trail_label shape->(labels(valence, arousal, dominance, liking))

            # only use the eeg electrodes data and
            # remove the pretrail data i.e., 3 rest seconds
            sub_data_list.append(trail_data[:32, pretrail:])
            sub_label_list.append(trail_label)

        # sub_data_list -> (trail, electrodes, raw_data)
        # sub_label_list -> (trail, labels)
        data[0].append(sub_data_list)
        label[0].append(sub_label_list)

    # data -> (session(1), subject, trail, electrode, raw_data)
    # label -> (session(1), subject, trail, labels)
    logger.debug(
        "Loaded DEAP data shape {}, label shape {}",
        np.array(data).shape,
        np.array(label).shape,
    )

    return data, label, 128, 32
