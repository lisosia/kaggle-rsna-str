import pandas as pd

train = pd.read_csv("train.csv")
portion = train.groupby("StudyInstanceUID")["pe_present_on_image"].mean()

portion.reset_index().to_csv("study_pos_portion.csv", index=False)
