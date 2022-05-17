from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import random
import winsound

import numpy as np
import os

import pandas
import pandas as pd
import mne

from sklearn.model_selection import train_test_split

def generate_adj_mx(df):
    df_matrix = pd.DataFrame(columns=["1", "2"])
    # replace 22 with 25 to revert
    for i in range(0, 22):
        for j in range(0, 22):
            df_matrix.loc[len(df_matrix.index)] = [i, j]

    # df.to_csv("matrix.csv", index=False)

    df_coord = pd.read_csv("coord.csv", header=None)

    df_matrix["3"] = 0.0
    for index, row in df_matrix.iterrows():
        # if row[0] != 25 and row[1] != 25:
        x1 = df_coord.loc[df_coord[0] == row[0], 1].iloc[0]
        y1 = df_coord.loc[df_coord[0] == row[0], 2].iloc[0]
        x2 = df_coord.loc[df_coord[0] == row[1], 1].iloc[0]
        y2 = df_coord.loc[df_coord[0] == row[1], 2].iloc[0]
        sum = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # df_matrix.at[index, "3"] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        distance = 2
        if sum > distance:
            df_matrix.at[index, "3"] = 0
        elif sum == 0:
            df_matrix.at[index, "3"] = 1
        else:
            df_matrix.at[index, "3"] = (distance - sum) / distance
        # elif row[0] == 25 and row[1] == 25:
        #     df_matrix.at[index, "3"] = 1
    df_matrix = df_matrix.pivot(index='1', columns='2')
    a = df_matrix.to_numpy()

    list_col = list(df)
    dict_col_int = {}
    i = 0
    for elem in list_col:
        dict_col_int[elem] = i
        i += 1
    filename = 'data/sensor_graph/adj_mx_bci22.pkl'
    fileObject = open(filename, 'wb')
    pickle.dump([list_col, dict_col_int, a], fileObject)
    fileObject.close()

    pickle_file = "data/sensor_graph/adj_mx.pkl"
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise

    pickle_file_1 = "data/sensor_graph/adj_mx_bci.pkl"
    try:
        with open(pickle_file_1, 'rb') as f:
            pickle_data_1 = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file_1, 'rb') as f:
            pickle_data_1 = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file_1, ':', e)
        raise


def generate_graph_seq2seq_io_data(
        # """indexa,"""
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    df = df.reset_index(drop=True)
    idx = df.index[(df['event'] >= 7) & (df['event'] <= 10)].tolist()
    events = df[['event']].copy()
    d = {0: [0, 0, 0, 0, 1], 7: [1, 0, 0, 0, 0], 8: [0, 1, 0, 0, 0], 9: [0, 0, 1, 0, 0], 10: [0, 0, 0, 1, 0]}
    events.event = events.event.map(d)
    df = df.drop(['event'], axis=1)

    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x = []
    y = []

    seq_len = args.seq_length_x
    count_after_event = int(seq_len)
    quarter_for_ev_zero = int(count_after_event / 4)
    zero_before = int(quarter_for_ev_zero / 2)
    zero_after = quarter_for_ev_zero - zero_before

    counter_even_odd = 1
    for i in idx:
        while zero_before > 0:
            x.append(data[i - zero_before - seq_len:i - zero_before])
            y.append([0, 0, 0, 0, 1])
            zero_before -= 1
        if counter_even_odd % 2 == 1:
            zero_before = int(quarter_for_ev_zero)
        else:
            zero_before = 0
        while count_after_event > 0:
            x.append(data[i - seq_len + count_after_event:i + count_after_event])
            y.append(events["event"].iloc[i])
            count_after_event -= 1
        count_after_event = int(seq_len)
        while zero_after > 0:
            x.append(data[i + zero_after:i + zero_after + seq_len])
            y.append([0, 0, 0, 0, 1])
            zero_after -= 1
        if counter_even_odd % 2 == 1:
            zero_after = 0
        else:
            zero_after = int(quarter_for_ev_zero)
        counter_even_odd += 1

    x = np.stack(x, axis=0)
    asdfsadf = pd.DataFrame(y, columns=["0", "1", "2", "3", "4"])
    print(asdfsadf.groupby(["0", "1", "2", "3", "4"]).size())
    return x, y


def prep_df(df):
    # event_dict = {'IdleEEG eyes open': 276, 'IdleEEG eyes closed': 277, 'Start of trial': 768,
    #               'LH': 769, 'RH': 770, 'FT': 771, 'Ton':772, 'Unkn':783, 'Rej':1023, 'Eye mov':1072, "STaRT": 32766}
    event_dict = {'IdleEEG eyes open': 3, 'IdleEEG eyes closed': 4, 'Start of trial': 6,
                  'LH': 7, 'RH': 8, 'FT': 9, 'Ton': 10, 'Rej': 1, 'Eye mov': 2, "STaRT": 5}

    df.drop(df.columns[24], axis=1, inplace=True)
    df.drop(df.columns[23], axis=1, inplace=True)
    df.drop(df.columns[22], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    return df


def load_all_experiments():
    # TODO deprecated stuff
    app_df = pandas.DataFrame()
    for i in range(1, 10):
        if i == 4:
            continue
        print(i)
        raw = mne.io.read_raw_gdf("Data Bci competition/A0" + str(i) + "T.gdf")
        temp_df = raw.to_data_frame(time_format=None)
        events = mne.events_from_annotations(raw)
        # TODO replace events id with eventcode
        events = events[0]
        df_events = pd.DataFrame(np.squeeze(events), columns=['row', 'B', 'event'])
        df_events.drop('B', axis=1, inplace=True)
        df_events = df_events.loc[(df_events['event'] >= 7) & (df_events['event'] <= 10)]

        # raw.plot_psd(fmax=50) #eeg db
        # raw.plot(duration=5, n_channels=25) #channels freq
        # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'],
        #                     first_samp=raw.first_samp) # events over time

        temp_df["event"] = 0
        for index, row in df_events.iterrows():
            a = row[0]
            b = row[1]
            temp_df.at[a, "event"] = b
        temp_df.drop(temp_df.columns[0], axis=1, inplace=True)
        app_df = app_df.append(temp_df)
    app_df = prep_df(app_df)
    return app_df


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = load_all_experiments()
    generate_adj_mx(df)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))

    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_whole_exp_training(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        # add_time_in_day=False,
        # add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", len(y))
    # Write the data into npz file.
    num_samples = x.shape[0]
    y = np.asarray(y)
    # TODO maybe y = all exp if it takes little time to test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1, stratify=y)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.3, shuffle=False)
    asdfsadf = pd.DataFrame(y_train, columns=["0", "1", "2", "3", "4"])
    print(asdfsadf.groupby(["0", "1", "2", "3", "4"]).size())
    for cat in ["train", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", len(_y))
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}" + args.name_extra + ".npz"),
            x=_x,
            y=_y,
        )


def generate_graph_seq2seq_whole_exp_training(df, x_offsets, y_offsets):
    x = []
    y = []
    ev_int = 49
    rest_int = 62
    events_df = df[['event']].copy()
    idx = df.index[(events_df['event'] >= 7) & (events_df['event'] <= 10)].tolist()
    for i in idx:
        a = events_df['event'].iloc[i]
        events_df.loc[(events_df.index >= i - ev_int) & (events_df.index <= i + ev_int), 'event'] = a
        events_df.loc[(events_df.index == i - rest_int - 1), 'event'] = 66
        events_df.loc[(events_df.index == i + rest_int + 1), 'event'] = 66
    new_df = df.iloc[
        np.unique(np.concatenate(
            [np.arange(max(i - rest_int - 1, 0), min(i + rest_int + 1, len(df))) for i in
             idx]))]
    new_df = new_df.reset_index(drop=True)
    new_events_df = events_df.iloc[
        np.unique(np.concatenate(
            [np.arange(max(i - rest_int - 1, 0), min(i + rest_int + 1, len(events_df))) for i in
             idx]))]
    new_events_df = new_events_df.reset_index(drop=True)
    d = {0: [0, 0, 0, 0, 1], 7: [1, 0, 0, 0, 0], 8: [0, 1, 0, 0, 0], 9: [0, 0, 1, 0, 0], 10: [0, 0, 0, 1, 0], 66: 66}
    new_events_df.event = new_events_df.event.map(d)
    new_df = new_df.drop(['event'], axis=1)
    data = np.expand_dims(new_df.values, axis=-1)

    num_samples, num_nodes = new_df.shape
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

    zeros = []
    changed = True
    before = True
    after = False
    # balance data by randomnly selecting from the "no movement" slices
    for t in range(min_t, max_t):  # t is the index of the last observation.
        slice_events = new_events_df[t + x_offsets[0]:t + 1]
        if 66 in slice_events.values:
            if not changed:
                changed = True
                before = not before
                after = not after
            continue
        if new_events_df["event"].iloc[t + x_offsets[0]] == [0, 0, 0, 0, 1]:
            changed = False
            if before:
                zeros.append(data[t + x_offsets, ...])
        elif new_events_df["event"].iloc[t] == [0, 0, 0, 0, 1]:
            changed = False
            if after:
                zeros.append(data[t + x_offsets, ...])
        else:
            changed = False
            x.append(data[t + x_offsets, ...])
            y.append(new_events_df["event"].iloc[t])
    quarter_data_size = int(len(y) / 4)
    zeros = random.sample(zeros, quarter_data_size)

    x.extend(zeros)
    l = [[0, 0, 0, 0, 1]] * quarter_data_size
    y.extend(l)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def load_all_experiments_list():
    list_exp = []
    for i in range(1, 10):
        if i == 4:
            continue
        print(i)
        raw = mne.io.read_raw_gdf("Data Bci competition/A0" + str(i) + "T.gdf")
        temp_df = raw.to_data_frame(time_format=None)
        events = mne.events_from_annotations(raw)
        # # TODO replace events id with eventcode
        events = events[0]
        df_events = pd.DataFrame(np.squeeze(events), columns=['row', 'B', 'event'])
        df_events.drop('B', axis=1, inplace=True)
        df_events = df_events.loc[(df_events['event'] >= 7) & (df_events['event'] <= 10)]
        temp_df["event"] = 0
        for index, row in df_events.iterrows():
            a = row[0]
            b = row[1]
            temp_df.at[a, "event"] = b
        temp_df.drop(temp_df.columns[0], axis=1, inplace=True)
        temp_df = prep_df(temp_df)
        list_exp.append(temp_df)
    return list_exp


def create_data_for_testing(args):
    list_experiments = load_all_experiments_list()
    index_exp = 1
    for df in list_experiments:
        seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
        x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
        y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
        x, y = generate_graph_seq2seq_whole_exp(
            df,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
        )
        index = str(index_exp)
        y = np.asarray(y)
        print(index, "x: ", x.shape, "y:", len(y))
        np.savez_compressed(
            os.path.join(args.output_dir + "/testing/", index + "whole_exp_testing.npz"),
            x=x,
            y=y,
        )
        if index_exp != 3:
            index_exp += 1
        else:
            index_exp += 2


def generate_graph_seq2seq_whole_exp(df, x_offsets, y_offsets):
    x = []
    y = []
    ev_int = 49
    rest_int = 62
    events_df = df[['event']].copy()
    idx = df.index[(events_df['event'] >= 7) & (events_df['event'] <= 10)].tolist()
    for i in idx:
        a = events_df['event'].iloc[i]
        events_df.loc[(events_df.index >= i - ev_int) & (events_df.index <= i + ev_int), 'event'] = a
        events_df.loc[(events_df.index == i - rest_int - 1), 'event'] = 66
        events_df.loc[(events_df.index == i + rest_int + 1), 'event'] = 66
    new_df = df.iloc[
        np.unique(np.concatenate(
            [np.arange(max(i - rest_int - 1, 0), min(i + rest_int + 1, len(df))) for i in
             idx]))]
    new_df = new_df.reset_index(drop=True)
    new_events_df = events_df.iloc[
        np.unique(np.concatenate(
            [np.arange(max(i - rest_int - 1, 0), min(i + rest_int + 1, len(events_df))) for i in
             idx]))]
    new_events_df = new_events_df.reset_index(drop=True)

    d = {0: [0, 0, 0, 0, 1], 7: [1, 0, 0, 0, 0], 8: [0, 1, 0, 0, 0], 9: [0, 0, 1, 0, 0], 10: [0, 0, 0, 1, 0], 66: 66}
    new_events_df.event = new_events_df.event.map(d)
    new_df = new_df.drop(['event'], axis=1)
    data = np.expand_dims(new_df.values, axis=-1)

    num_samples, num_nodes = new_df.shape
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        slice_events = new_events_df[t + x_offsets[0]:t + 1]
        if 66 in slice_events.values:
            continue
        if new_events_df["event"].iloc[t] == [0, 0, 0, 0, 1] or \
                new_events_df["event"].iloc[t + x_offsets[0]] == [0, 0, 0, 0, 1]:
            x.append(data[t + x_offsets, ...])
            y.append([0, 0, 0, 0, 1])
        else:
            x.append(data[t + x_offsets, ...])
            y.append(new_events_df["event"].iloc[t])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/BCI", help="Output directory.")
    # 0.004s
    parser.add_argument("--seq_length_x", type=int, default=50, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=50, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--name_extra", type=str, default = "50_62_0before_or_after_7030")
    parser.add_argument("--dow", action='store_true', )
    creating_testing = False

    args = parser.parse_args()
    # if os.path.exists(args.output_dir):
    #     reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.output_dir)
    if not creating_testing:
        generate_train_val_test(args)
    else:
        create_data_for_testing(args)

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
