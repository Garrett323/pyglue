import pandas as pd
from pandas.testing import assert_frame_equal
from pyglue import Encoder


def test_encode_decode_mixed_dataframe():
    df = pd.DataFrame({
        "age": [25, 31, 42, 28],
        "height": [1.75, 1.82, 1.68, 1.91],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "city": ["Berlin", "Paris", "Berlin", "London"],
    })

    encoder = Encoder()

    encoded = encoder.encode(df)
    decoded = encoder.decode(encoded)

    assert_frame_equal(decoded, df, check_dtype=False)
