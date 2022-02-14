from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle

import numpy as np
import os

import pandas
import pandas as pd
import mne


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
        if sum == 0:
            df_matrix.at[index, "3"] = 1
        elif sum == 1:
            df_matrix.at[index, "3"] = 0.5
        elif sum == 2:
            df_matrix.at[index, "3"] = 0.25

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


def get_events_indexes_cut_windows(df):
    df = df.reset_index(drop=True)
    idx = df.index[(df['event'] >= 7) & (df['event'] <= 10)].tolist()
    n = 20
    print(len(df))
    new_df = df.iloc[np.unique(np.concatenate([np.arange(max(i - n, 0), min(i + (n / 4) + 1, len(df))) for i in idx]))]
    new_df = new_df.reset_index(drop=True)
    idx = new_df.index[(new_df['event'] >= 7) & (new_df['event'] <= 10)].tolist()

    events = []
    for index in idx:
        a = new_df.iloc[[index]]
        a = a.iat[0, 25]
        events.append(a)
    final_df = pd.DataFrame(columns=['idx', 'event'])
    counter = 0
    for i in range(0, len(idx)):
        for j in range(20, -6,-1):
            final_df.loc[counter] = [idx[i]-j, events[i]]
            counter += 1

        # if indexa == 1:
    final_df.to_csv("allExpIE.csv", index=False)
    return new_df


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
    # indexa
    # df, events = cut_windows(df,20,5)

    num_samples, num_nodes = df.shape
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
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def prep_df(df):
    # df = raw.to_data_frame(time_format=None)
    # event_dict = {'IdleEEG eyes open': 276, 'IdleEEG eyes closed': 277, 'Start of trial': 768,
    #               'LH': 769, 'RH': 770, 'FT': 771, 'Ton':772, 'Unkn':783, 'Rej':1023, 'Eye mov':1072, "STaRT": 32766}
    event_dict = {'IdleEEG eyes open': 3, 'IdleEEG eyes closed': 4, 'Start of trial': 6,
                  'LH': 7, 'RH': 8, 'FT': 9, 'Ton': 10, 'Rej': 1, 'Eye mov': 2, "STaRT": 5}

    df = get_events_indexes_cut_windows(df)
    df.drop(df.columns[25], axis=1, inplace=True)

    df.drop(df.columns[24], axis=1, inplace=True)
    df.drop(df.columns[23], axis=1, inplace=True)
    df.drop(df.columns[22], axis=1, inplace=True)

    return df


def load_all_experiments():
    app_df = pandas.DataFrame()
    for i in range(1, 10):
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
    return app_df


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    # raw = mne.io.read_raw_gdf("Data Bci competition/A01T.gdf")
    # raw_T = mne.io.read_raw_gdf("Data Bci competition/A02T.gdf")
    app_df = load_all_experiments()
    df = prep_df(app_df)
    generate_adj_mx(df)

    # df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    # df1 = prep_df(raw)
    # indexa = 1
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=args.dow,
        # indexa=indexa,
    )
    # indexa += 1
    # x_t, y_t = generate_graph_seq2seq_io_data(
    #     df1,
    #     x_offsets=x_offsets,
    #     y_offsets=y_offsets,
    #     add_time_in_day=False,
    #     add_day_in_week=args.dow,
    #     indexa=indexa,
    # )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    # x_train, y_train = x, y
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    # x_test, y_test = x_t, y_t

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}22.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/BCI", help="Output directory.")
    # parser.add_argument("--traffic_df_filename", type=str, default="data/metr-la.h5",
    #                     help="Raw traffic readings.", )
    # parser.add_argument("--traffic_df_filename", type=str, default="Data Bci competition/A01T.gdf",
    #                     help="Raw traffic readings.", )
    # parser.add_argument("--traffic_df_filename1", type=str, default="Data Bci competition/A02T.gdf",
    #                     help="Raw traffic readings.", )
    # parser.add_argument("--index_event_output", type=str, default="A01Tie.csv",
    #                     help="Index event.", )
    # parser.add_argument("--index_event_output1", type=str, default="A02Tie.csv",
    #                     help="Index event.", )
    # TODO 25 maybe 0.004s *25 = 0.1
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true', )

    args = parser.parse_args()
    # if os.path.exists(args.output_dir):
    #     reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.output_dir)
    generate_train_val_test(args)
