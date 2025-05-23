import pandas as pd

df = pd.read_csv("results/open-world/MObI_all-classes_256/realism_table.csv")
order = ["id-ref", "track-ref", "in-domain-ref", "cross-domain-ref"]

# Sort and select data
df = df.drop_duplicates(subset="Reference Type")
df_sorted = df.set_index("Reference Type").loc[order].reset_index()

# Round values and join them as a string
result = " & ".join(
    list(
        map(
            str,
            df_sorted[["FID", "LPIPS", "CLIP"]].round({"FID": 2, "LPIPS": 3, "CLIP": 2}).values.reshape(-1)
        )
    )
)

print(result)

return

# --------------------------

import pandas as pd

df = pd.read_csv("results/open-world/copy-paste/realism_table.csv")

# Calculate average for reinsertion (id-ref + track-ref) and replacement (in-domain-ref + cross-domain-ref)
reinsertion_avg = df[df["Reference Type"].isin(["id-ref", "track-ref"])][["FID", "LPIPS", "CLIP", "FRD"]].mean()
replacement_avg = df[df["Reference Type"].isin(["in-domain-ref", "cross-domain-ref"])][["FID", "LPIPS", "CLIP", "FRD"]].mean()

# Create a new DataFrame with only the average rows
avg_df = pd.DataFrame({
    "Reference Type": ["reinsertion_avg", "replacement_avg"],
    "FID": [reinsertion_avg["FID"], replacement_avg["FID"]],
    "LPIPS": [reinsertion_avg["LPIPS"], replacement_avg["LPIPS"]],
    "CLIP": [reinsertion_avg["CLIP"], replacement_avg["CLIP"]],
    "FRD": [reinsertion_avg["FRD"], replacement_avg["FRD"]]
})

# Round values and join them as a string
result = " & ".join(
    list(
        map(
            str,
            avg_df[["FID", "LPIPS", "CLIP", "FRD"]].round({"FID": 2, "LPIPS": 3, "CLIP": 2, "FRD": 3}).values.reshape(-1)
        )
    )
)

print(result)

# --------------------------

import pandas as pd

df = pd.read_csv("results/open-world/copy-paste/realism_table.csv")

# Calculate average for reinsertion (id-ref + track-ref) and replacement (in-domain-ref + cross-domain-ref) without FRD
reinsertion_avg = df[df["Reference Type"].isin(["id-ref", "track-ref"])][["FID", "LPIPS", "CLIP"]].mean()
replacement_avg = df[df["Reference Type"].isin(["in-domain-ref", "cross-domain-ref"])][["FID", "LPIPS", "CLIP"]].mean()

# Create a new DataFrame with only the average rows (without FRD)
avg_df = pd.DataFrame({
    "Reference Type": ["reinsertion_avg", "replacement_avg"],
    "FID": [reinsertion_avg["FID"], replacement_avg["FID"]],
    "LPIPS": [reinsertion_avg["LPIPS"], replacement_avg["LPIPS"]],
    "CLIP": [reinsertion_avg["CLIP"], replacement_avg["CLIP"]]
})

# Round values and join them as a string
result = " & ".join(
    list(
        map(
            str,
            avg_df[["FID", "LPIPS", "CLIP"]].round({"FID": 2, "LPIPS": 3, "CLIP": 2}).values.reshape(-1)
        )
    )
)

print(result)

# --------------------------

import pandas as pd
import os

path = "results/final_results/MObI_512_epoch28"

df = pd.read_csv(os.path.join(path, "realism_table.csv"))

reinsertion_avg = df[df["Reference Type"].isin(["id-ref", "track-ref"])][["FID", "LPIPS", "CLIP", "D-LPIPS", "I-LPIPS"]].mean()
replacement_avg = df[df["Reference Type"].isin(["in-domain-ref", "cross-domain-ref"])][["FID", "LPIPS", "CLIP", "D-LPIPS", "I-LPIPS"]].mean()

avg_df = pd.DataFrame({
    "Reference Type": ["reinsertion_avg", "replacement_avg"],
    "FID": [reinsertion_avg["FID"], replacement_avg["FID"]],
    "LPIPS": [reinsertion_avg["LPIPS"], replacement_avg["LPIPS"]],
    "CLIP": [reinsertion_avg["CLIP"], replacement_avg["CLIP"]],
    "D-LPIPS": [reinsertion_avg["D-LPIPS"], replacement_avg["D-LPIPS"]],
    "I-LPIPS": [reinsertion_avg["I-LPIPS"], replacement_avg["I-LPIPS"]]
})

# Round values and join them as a string
result = f"{path} & " + " & ".join(
    list(
        map(
            str,
            avg_df[["FID", "LPIPS", "CLIP", "D-LPIPS", "I-LPIPS"]].round({"FID": 2, "LPIPS": 3, "CLIP": 2, "D-LPIPS": 3, "I-LPIPS": 3}).values.reshape(-1)
        )
    )
)
print(result)