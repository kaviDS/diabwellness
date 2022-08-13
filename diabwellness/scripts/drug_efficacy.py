# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved

from typing import Optional

import pandas as pd
from scipy import stats
from diabwellness.utils.data_utils import dict_partial_match, DRUG_CLASSES


class DrugEfficacy:
    def __init__(
        self,
        meas_df: pd.DataFrame,
        pres_df: pd.DataFrame,
        drug_df: pd.DataFrame,
    ) -> None:
        self.meas_df = meas_df
        self.pres_df = pres_df
        self.drug_df = drug_df
        self.efficacy_df: Optional[pd.DataFrame] = None

    def map_generic_names(self):
        # extract relevant columns:
        self.pres_df = self.pres_df[
            [
                "APPOINT_ID",
                "DRUG_NAME",
                "DRUG_TYPE",
                "ITEM_ID",
                "PATIENT_ID",
                "QTY",
                "PHARMACY_DRUG_ID",
            ]
        ]
        # remove single dose insulin prescriptions:
        self.pres_df = (
            self.pres_df[
                ~(
                    (
                        self.pres_df["DRUG_NAME"].str.contains(
                            "lantus|tresiba|basalog", case=False, regex=True
                        )
                    )
                    & (self.pres_df["QTY"] == 0)
                )
            ]
            .dropna()
            .reset_index(drop=True)
        )
        # self.pres_df = self.pres_df.astype({"PATIENT_ID": 'int'}, errors = 'ignore')
        self.pres_df["PATIENT_ID"] = self.pres_df["PATIENT_ID"].astype(int)
        # extract relevant columns:
        self.drug_df = (
            self.drug_df[["DRUG_ID", "DRUG_NAME", "GENERIC_NAME"]]
            .dropna()
            .reset_index(drop=True)
        )

        # map drug names to generic names based on drug id:
        # self.drug_df['DRUG_ID'] and self.pres_df['ITEM_ID'] are the same:
        drug_master = self.drug_df.set_index("DRUG_ID")["GENERIC_NAME"]
        self.pres_df["GENERIC_NAME"] = self.pres_df["ITEM_ID"].replace(drug_master)

        return self.pres_df

    def map_drug_classes(self):
        # map generic names to drug class combinations:
        drug_classes = dict_partial_match(DRUG_CLASSES)
        self.pres_df["DRUG_CLASS"] = self.pres_df["GENERIC_NAME"].apply(
            lambda x: drug_classes[x]
        )
        return self.pres_df.dropna(subset=["DRUG_CLASS"]).reset_index(drop=True)

    def extract_drug_combinations(self):
        # collect all anti-diabetic drugs across multiple tablets in the first appointment:
        def combined_list(input_list):
            # return tuple(sorted(set([a for b in input_list for a in b])))
            return tuple(sorted(set().union(*input_list)))

        self.pres_df = (
            self.pres_df.groupby("PATIENT_ID").apply(
                lambda x: (
                    x.groupby(["APPOINT_ID"]).apply(
                        lambda y: combined_list(y["DRUG_CLASS"].to_list())
                    )
                ).iloc[0]
            )
        ).to_frame(name="DRUG_COMBINATION")

        # number of drugs in each combination:
        self.pres_df["DRUG_COUNT"] = self.pres_df["DRUG_COMBINATION"].apply(
            lambda x: len(x)
        )
        self.pres_df.index.name = "NFID"
        return self.pres_df

    def calculate_efficacy(self):
        # combine the drug combinations and A1C values, count the drugs:
        self.efficacy_df = (
            pd.concat([self.pres_df, self.meas_df], axis=1)
            .dropna()
            .groupby("DRUG_COMBINATION")
            .apply(
                lambda x: [
                    x["A1C_1"].count(),
                    (x["NEW_T2DM"] == True).sum(),
                    x["A1C_1"].mean(),
                    x["A1C_2"].mean(),
                    (x["A1C_2"] - x["A1C_1"]).mean(),
                ]
                + list(stats.ttest_ind(a=x.A1C_1, b=x.A1C_2, equal_var=False))
            )
        )
        self.efficacy_df = pd.DataFrame(
            self.efficacy_df.tolist(),
            index=self.efficacy_df.index,
            columns=[
                "patient_counts",
                "new_t2dm_counts",
                "mean_a1c_first",
                "mean_a1c_second",
                "mean_a1c_reduction",
                "z_statistics",
                "p_value",
            ],
        )
        return self.efficacy_df

    def extract_biguanide_combinations(self, drug_counts=2):
        if self.efficacy_df:
            self.efficacy_df = self.efficacy_df.loc[
                (self.efficacy_df.index.map(len) == drug_counts)
                & (
                    self.efficacy_df.index.to_series().apply(
                        lambda x: True if "Biguanide" in x else False
                    )
                )
            ]
            # sort according to p-values:
            counts_cond = self.efficacy_df["patient_counts"] > 0
            return self.efficacy_df.loc[counts_cond].sort_values(
                ["p_value"], ascending=True
            )
