import pandas as pd

df = pd.read_csv("merged_eval_outputs_filtered.csv")
print(len(df))
df = df[df['N'] != 0]
df = df[df['rating'].isin([5])]
print(len(df))
# 🔀 Shuffle entire dataframe first
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Columns to stratify on
features = ["MODEL_ID", "SOURCE", "BASE", "rating"]

# Create groups
groups = df.groupby(features)

# Number of groups
num_groups = groups.ngroups

# Compute how many per group
per_group = 2250 // num_groups
print("Groups:", num_groups, "Per group:", per_group)

# Sample uniformly from each group
sampled = (
    groups.apply(lambda g: g.sample(min(len(g), per_group), random_state=42))
          .reset_index(drop=True)
)

print("Final sample size:", len(sampled))

sampled.to_csv("stratified_2250_uniform.csv", index=False)
