import pandas as pd
import numpy as np

years = [2019, 2020, 2021, 2022, 2023, 2024]
OH_nh = [
    0.931052793,
    0.920650489,
    0.914930976,
    0.911705478,
    0.910246531,
    0.905414623
    ]
OH_sh = [
    0.983772452,
    0.979142384,
    0.973963259,
    0.971819293,
    0.968996907,
    0.96737343
]
df = pd.DataFrame(
    data={'OH_nh': OH_nh,
          'OH_sh': OH_sh},
          index=years
          )

df.to_csv('figures/data/OH_SF.csv')