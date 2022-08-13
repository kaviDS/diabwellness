# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved
 
from collections import Counter
import itertools
import pandas as pd
from diabwellness.utils.data_utils import COMPLICATION_TYPES, PATIENT_TYPES


class PatientClustering:
    def __init__(self) -> None:
        pass

    def cluster_aggregator(self, group):
        output_dict = {}

        def extract_set_fractions(series, types_dict):
            counts_dict = Counter(
                {given_type: 0 for given_type in set(types_dict.values())}
            )
            cluster_list = series.dropna().apply(lambda x: list(x)).tolist()
            merged_list = list(itertools.chain(*cluster_list))
            counts_dict.update(merged_list)
            return {
                f"{series.name}_{k}": f"{v}/{series.size}"
                for k, v in counts_dict.items()
            }

        # columns where average value should be taken
        mean_cols = [
            "HEIGHT",
            "WEIGHT",
            "BMI",
            "BP",
            "DIA_BP",
            "PULSE",
            "DURATION",
            "AGE",
        ]
        output_dict = {col: group[col].mean() for col in mean_cols}

        # count of new T2DM patients in each cluster
        output_dict["NEW_T2DM"] = group["NEW_T2DM"].sum()

        # first, second and last values of A1C values are considered
        output_dict["A1C_first"] = (
            group["A1C_first_two_values"].apply(lambda x: x[0]).mean()
        )
        output_dict["A1C_second"] = (
            group["A1C_first_two_values"].apply(lambda x: x[1]).mean()
        )
        output_dict["A1C_last"] = (
            group["A1C"].apply(lambda x: pd.Series(x).dropna().iloc[-1]).mean()
        )

        # get the size of patients in each cluster:
        output_dict["CLUSER_SIZE"] = group.shape[0]

        # first and last values of FF and PP  are considered
        #     print(group['FS'])
        #     output_dict['FS_first'] = group['FS'].apply(lambda x: pd.Series(x).dropna().iloc[0]).mean()
        #     output_dict['FS_last'] = group['FS'].apply(lambda x: pd.Series(x).dropna().iloc[-1]).mean()
        #     output_dict['PP_first'] = group['PP'].apply(lambda x: pd.Series(x).dropna().iloc[0]).mean()
        #     output_dict['PP_last'] = group['PP'].apply(lambda x: pd.Series(x).dropna().iloc[-1]).mean()

        # combine COMPLICATIONS, PATIENT_TYPE into a single set and then to proportions:
        output_dict.update(
            extract_set_fractions(group["COMPLICATIONS"], COMPLICATION_TYPES)
        )
        output_dict.update(extract_set_fractions(group["PATIENT_TYPE"], PATIENT_TYPES))

        return pd.Series(output_dict)
