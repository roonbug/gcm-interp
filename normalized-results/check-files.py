import pandas as pd

df = pd.read_csv('judge_prompts.csv')

# Column to inspect
col = df["judge_prompt"]

# Boolean mask of rows where judge_prompt is NOT a string
mask = col.map(lambda x: not isinstance(x, str))

# Extract the problematic rows
non_string_rows = df[mask]

if not non_string_rows.empty:
    print("Non-string judge_prompt values found.\n")
    print(non_string_rows.to_string(index=False))
else:
    print("All values are strings.")
