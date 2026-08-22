import pandas as pd
import pyperf
from pyglue import Encoder

rows = 100_000
frame = pd.DataFrame({
      "number": range(rows),
      "category": [f"group-{i % 100}" for i in range(rows)],
})

encoder = Encoder()
encoded = encoder.encode(frame)[0]
runner = pyperf.Runner(processes=40, values=10)
runner.bench_func("encode-mixed-100k", encoder.encode, frame)
runner.bench_func("decode-mixed-100k", encoder.decode, encoded)
