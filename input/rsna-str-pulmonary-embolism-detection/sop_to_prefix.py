from glob import glob
from tqdm import tqdm
import os
import pandas as pd

df_data = []
for jpg in tqdm(sorted(glob("train-jpegs/*/*/*.jpg"))):
    base = os.path.basename(jpg)[:-4]
    prefix = base[:4]
    sop = base[5:]
    df_data.append({"SOPInstanceUID": sop, "img_prefix": prefix})

df = pd.DataFrame(df_data)
df.to_csv("sop_to_prefix.csv", index=False)   

