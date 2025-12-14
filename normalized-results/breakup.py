import pandas as pd
import math

input_csv = "relevance_fluency_prompts.csv"

df = pd.read_csv(input_csv)

# Number of splits
n = 4
chunk_size = math.ceil(len(df) / n)

for i in range(n):
    chunk = df[i * chunk_size : (i + 1) * chunk_size]
    chunk.to_csv(f"rf_{i}.csv", index=False)

print("Done!")
