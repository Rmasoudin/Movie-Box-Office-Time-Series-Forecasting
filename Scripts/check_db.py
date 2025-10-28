import pandas as pd

original = pd.read_csv("data/movies.csv", low_memory=False)
exported = pd.read_csv("data/exported_movies.csv", low_memory=False)
original = original.drop('index', axis = 1)
original = original.iloc[ : , 1 :]
for df in [original, exported]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
# original = original.iloc[1 : , :].copy()
# original = original.reindex(sorted(original.columns), axis=1)
# exported = exported.reindex(sorted(exported.columns), axis=1)
original.reset_index(drop=True, inplace=True)
exported.reset_index(drop=True, inplace=True)
print("Exact match:", original.equals(exported))
if not original.equals(exported):
    diff = original.compare(exported)
    print("\nDifferences found:")
    print(diff)