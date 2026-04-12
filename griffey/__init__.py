# import os
# import logging
# import ast
# import pandas as pd
# import re

# logger = logging.getLogger(__name__)

# EVENTS = {
#     0: "Address (A)",
#     1: "Toe-up (TU)",
#     2: "Mid-backswing (MB)",
#     3: "Top (T)",
#     4: "Mid-downswing (MD)",
#     5: "Impact (I)",
#     6: "Mid-follow-through (MFT)",
#     7: "Finish (F)"
# }


# class LocalBlobfs:
#     def __init__(self):
#         self.root_dir = os.environ.get("BLOB_PREFIX", os.getcwd())
#         logger.info(f"Initialized LocalBlobfs with root directory: {self.root_dir}")

#     def ls(self) -> list:
#         return os.listdir(self.root_dir)

#     def read_csv(self, key: str) -> pd.DataFrame:
#         return pd.read_csv(os.path.join(self.root_dir, key))

#     def write_csv(self, df: pd.DataFrame, key: str) -> None:
#         df.to_csv(os.path.join(self.root_dir, key), index=False)


# def read_train_data(blobfs: LocalBlobfs) -> pd.DataFrame:
#     df_labels = blobfs.read_csv("GolfDB.csv")

#     for c in ["id", "Unnamed: 0"]:
#         if c in df_labels.columns:
#             df_labels.drop(columns=[c], inplace=True)

#     df_labels["events"] = df_labels["events"].apply(ast.literal_eval)
#     df_labels["bbox"] = df_labels["bbox"].apply(_parse_bbox_string)
#     return df_labels


# def _parse_bbox_string(s: str) -> list:
#     s_clean = s.strip()[1:-1]  # Remove the outer brackets
#     list_of_strings = re.split(r'[,\s]+', s_clean)  # Split by comma or whitespace
#     list_of_floats = [float(e) for e in list_of_strings if e]  # Convert to float and filter out empty strings
#     return list_of_floats


# if __name__ == "__main__":
#     blobfs = LocalBlobfs()
#     df_labels = read_train_data(blobfs)
#     print(df_labels.head())
#     print(f"Events: {df_labels.iloc[0]["events"]}")
#     print(f"Events shape: {len(df_labels.iloc[0]["events"])}")
#     print(f"BBox: {df_labels.iloc[0]["bbox"]}")
#     print(f"BBox shape: {len(df_labels.iloc[0]["bbox"])}")
