import pandas as pd
import pickle
import numpy as np
from pandas.testing import assert_frame_equal
from pyglue.pyglue import Encoder


def test_encode_decode_mixed_dataframe():
    df = pd.DataFrame({
        "age": [25, 31, 42, 28],
        "height": [1.75, 1.82, 1.68, 1.91],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "city": ["Berlin", "Paris", "Berlin", "London"],
    })
    encoder = Encoder()
    encoded, cat_cols = encoder.encode(df)
    print(encoded)
    decoded = encoder.decode(encoded)
    print(decoded)
    assert_frame_equal(decoded, df, check_dtype=False)
    assert(len(cat_cols) == 2)
    for i, j in zip(cat_cols,[2,3]):
        assert(i == j)


def test_encode_decode_all_numeric_dataframe():
    df = pd.DataFrame({
        "age": [25, 31, 42, 28],
        "height": [1.75, 1.82, 1.68, 1.91],
    })

    encoder = Encoder()
    encoded, cat_cols = encoder.encode(df)
    decoded = encoder.decode(encoded)

    assert cat_cols is None
    assert_frame_equal(decoded, df, check_dtype=False)


def test_encode_decode_all_categorical_dataframe():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "city": ["Berlin", "Paris", "Berlin", "London"],
    })

    encoder = Encoder()
    encoded, cat_cols = encoder.encode(df)
    decoded = encoder.decode(encoded)

    assert cat_cols == [0, 1]
    assert_frame_equal(decoded, df, check_dtype=False)


def test_encode_decode_all_numeric_numpy_array():
    arr = np.array([
        [25, 1.75],
        [31, 1.82],
        [42, 1.68],
        [28, 1.91],
    ])

    encoder = Encoder()
    encoded, cat_cols = encoder.encode(arr)
    decoded = encoder.decode(encoded)

    assert cat_cols is None
    np.testing.assert_array_equal(decoded, arr)


def test_encode_decode_all_categorical_numpy_array():
    arr = np.array([
        ["Alice", "Berlin"],
        ["Bob", "Paris"],
        ["Charlie", "Berlin"],
        ["Diana", "London"],
    ])

    encoder = Encoder()
    encoded, cat_cols = encoder.encode(arr)
    decoded = encoder.decode(encoded)

    assert cat_cols == [0, 1]
    np.testing.assert_array_equal(decoded, arr)


def test_picklable():
    enc = Encoder()
    data = pickle.dumps(enc)      # if this raises, pytest fails the test — good
    restored = pickle.loads(data)
    assert isinstance(restored, Encoder)
