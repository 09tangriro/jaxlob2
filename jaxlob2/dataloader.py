import os
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import py7zr
import tqdm

from jaxlob2 import job


def extract_zip(
    lobster_zip_path: str,
    ticker: str,
    date_range: Tuple[str, str],
    depth: int = 10,
    write_path: str = "../data/",
) -> None:
    """Extract the LOBSTER CSV files from the given .7z file.

    :param lobster_zip_path: Path to the LOBSTER .7z file.
    :param ticker: Ticker symbol.
    :param date_range: Tuple containing the start and end date in the format "YYYY-MM-DD".
    :param depth: Depth of the order book.
    :param write_path: Path to write the extracted files to.
    """
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq="B")
    dates = [date.strftime("%Y-%m-%d") for date in dates]
    targets = [
        f"{ticker}_{date}_34200000_57600000_orderbook_{depth}.csv" for date in dates
    ]
    targets += [
        f"{ticker}_{date}_34200000_57600000_message_{depth}.csv" for date in dates
    ]
    # Open the .7z file in read mode
    with py7zr.SevenZipFile(lobster_zip_path, "r") as archive:
        archive.extract(path=write_path, targets=targets)


def load_files(data_path: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Load all files in the given directory into separate lists for messages and orderbooks.

    :param data_path: Path to the directory containing the data files.

    :return messages: List of DataFrames containing the messages.
    :return orderbooks: List of DataFrames containing the orderbooks.
    """
    orderbooks = []
    messages = []

    # List all files in the directory
    files = sorted(os.listdir(data_path))
    dtype = {0: float, 1: int, 2: int, 3: int, 4: int, 5: int}

    # Iterate through each file
    for file in tqdm.tqdm(files, total=len(files), desc="Loading files"):
        if file.endswith(".csv"):
            if "orderbook" in file.lower():
                orderbooks.append(
                    pd.read_csv(os.path.join(data_path, file), header=None)
                )
            elif "message" in file.lower():
                messages.append(
                    pd.read_csv(
                        os.path.join(data_path, file),
                        dtype=dtype,
                        usecols=range(6),
                        header=None,
                    )
                )

    return messages, orderbooks


def split_timestamp(message: pd.DataFrame) -> pd.DataFrame:
    """Split the timestamp into seconds and nanoseconds."""
    message[6] = message[0].apply(lambda x: int(x))
    message[7] = ((message[0] - message[6]) * int(1e9)).astype(int)
    message.columns = [
        "time",
        "type",
        "order_id",
        "qty",
        "price",
        "direction",
        "time_s",
        "time_ns",
    ]
    return message


def filter_valid(message: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Filter out messages with invalid types."""
    message = message[message.type.isin([1, 2, 3, 4])]
    valid_index = message.index.to_numpy()
    message.reset_index(inplace=True, drop=True)
    return message, valid_index


def adjust_executions(message: pd.DataFrame) -> pd.DataFrame:
    """Adjust the direction and type of market order messages."""
    message.loc[message["type"] == job.MessageType.MARKET, "direction"] *= -1
    message.loc[message["type"] == job.MessageType.MARKET, "type"] = (
        job.MessageType.LIMIT
    )
    return message


def remove_deletes(message: pd.DataFrame) -> pd.DataFrame:
    """Remove delete messages from the dataset, replace with cancel."""
    message.loc[message["type"] == job.MessageType.DELETE, "type"] = (
        job.MessageType.CANCEL
    )
    return message


def add_agent_id(message: pd.DataFrame) -> pd.DataFrame:
    """Add an agent ID column to the messages."""
    import warnings

    from pandas.errors import SettingWithCopyWarning

    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    message["agent_id"] = job.MARKET_AGENT_ID
    return message


def load_cubes(
    data_path: str,
    start_time: int,
    end_time: int,
    episode_time: int,
    messages_per_step: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    messages, orderbooks = load_files(data_path)

    def preProcessingMassegeOB(message: pd.DataFrame, orderbook: pd.DataFrame):
        message = split_timestamp(message)
        message, valid_index = filter_valid(message)
        message = adjust_executions(message)
        message = remove_deletes(message)
        message = add_agent_id(message)
        orderbook.iloc[valid_index, :].reset_index(inplace=True, drop=True)
        return message, orderbook

    pairs = [
        preProcessingMassegeOB(message, orderbook)
        for message, orderbook in zip(messages, orderbooks)
    ]
    messages, orderbooks = zip(*pairs)

    indices = list(range(start_time, end_time, episode_time))

    def sliceWithoutOverlap(message: pd.DataFrame, orderbook: pd.DataFrame):
        def splitMessage(message: pd.DataFrame, orderbook: pd.DataFrame):
            sliced_parts = []
            init_OBs = []
            for i in range(len(indices) - 1):
                start_index = indices[i]
                end_index = indices[i + 1]
                index_s, index_e = (
                    message[
                        (message["time"] >= start_index) & (message["time"] < end_index)
                    ]
                    .index[[0, -1]]
                    .tolist()
                )
                index_e = (
                    index_e // messages_per_step + 10
                ) * messages_per_step + index_s % messages_per_step
                assert (index_e - index_s) % messages_per_step == 0, "wrong code 31"
                sliced_part = message.loc[np.arange(index_s, index_e)]
                sliced_parts.append(sliced_part)
                init_OBs.append(orderbook.iloc[index_s, :])

            # Last sliced part from last index to end_time
            start_index = indices[i]
            end_index = indices[i + 1]
            index_s, index_e = (
                message[
                    (message["time"] >= start_index) & (message["time"] < end_index)
                ]
                .index[[0, -1]]
                .tolist()
            )
            index_s = (
                index_s // messages_per_step - 10
            ) * messages_per_step + index_e % messages_per_step
            assert (index_e - index_s) % messages_per_step == 0, "wrong code 32"
            last_sliced_part = message.loc[np.arange(index_s, index_e)]
            sliced_parts.append(last_sliced_part)
            init_OBs.append(orderbook.iloc[index_s, :])
            for part in sliced_parts:
                # print("start")
                assert (
                    part.time_s.iloc[-1] - part.time_s.iloc[0] >= episode_time
                ), f"wrong code 33, {part.time_s.iloc[-1] - part.time_s.iloc[0]}, {episode_time}"
                assert part.shape[0] % messages_per_step == 0, "wrong code 34"
            return sliced_parts, init_OBs

        sliced_parts, init_OBs = splitMessage(message, orderbook)

        def sliced2cude(sliced):
            columns = [
                "type",
                "direction",
                "price",
                "qty",
                "agent_id",
                "order_id",
                "time_s",
                "time_ns",
            ]
            cube = sliced[columns].to_numpy()
            cube = cube.reshape((-1, messages_per_step, 8))
            return cube

        # def initialOrderbook():
        slicedCubes = [sliced2cude(sliced) for sliced in sliced_parts]
        # Cube: dynamic_horizon * stepLines * 8
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB

    slicedCubes_withOB_list = [
        sliceWithoutOverlap(message, orderbook)
        for message, orderbook in zip(messages, orderbooks)
    ]
    # i = 6 ; message,orderbook = messages[i],orderbooks[i]
    # slicedCubes_list(nested list), outer_layer : day, inter_later : time of the day

    def nestlist2flattenlist(nested_list):
        import itertools

        flattened_list = list(itertools.chain.from_iterable(nested_list))
        return flattened_list

    Cubes_withOB = nestlist2flattenlist(slicedCubes_withOB_list)

    max_steps_in_episode_arr = jnp.array(
        [m.shape[0] for m, _ in Cubes_withOB], jnp.int32
    )

    def Cubes_withOB_padding(Cubes_withOB):
        max_m = max(m.shape[0] for m, o in Cubes_withOB)
        new_Cubes_withOB = []
        for cube, OB in Cubes_withOB:

            def padding(cube, target_shape):
                pad_width = np.zeros((100, 8))
                # Calculate the amount of padding required
                padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
                padded_cube = np.pad(cube, padding, mode="constant", constant_values=0)
                return padded_cube

            cube = padding(cube, max_m)
            new_Cubes_withOB.append((cube, OB))
        return new_Cubes_withOB

    Cubes_withOB = Cubes_withOB_padding(Cubes_withOB)

    messages = [jnp.array(cube) for cube, _ in Cubes_withOB]
    books = [jnp.array(book) for _, book in Cubes_withOB]
    return jnp.array(messages), jnp.array(books), max_steps_in_episode_arr
