import pandas as pd

train = pd.read_csv("mitbih_train.csv", header=None)
label_col = train.columns[-1]
df_balanced_4000 = train.groupby(label_col).apply(lambda x: x.sample(n=4000, random_state=42, replace=True)).reset_index(drop=True)
df_balanced_4000.to_excel("ecg_4000_samples_each.xlsx", index=False)

print("Balanced dataset with 4000 samples per class saved")
print(df_balanced_4000[label_col].value_counts())