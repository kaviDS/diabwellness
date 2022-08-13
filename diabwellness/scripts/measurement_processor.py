# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved

from typing import Optional

import pandas as pd
import numpy as np

from diabwellness.utils.data_utils import (
    dict_partial_match,
    COMPLICATION_TYPES,
    PATIENT_TYPES,
)


class MeasurementProcessor:
    def __init__(self, meas_df: pd.DataFrame, pat_df: pd.DataFrame) -> None:
        self.meas_df = meas_df
        self.pat_df = pat_df
        self.meas_pat_df: Optional[pd.DataFrame] = None

    def preprocess_measurements(self):
        # preprocess the measurement details:
        self.meas_df = (
            self.meas_df[
                [
                    "APPOINT_ID",
                    "NFID",
                    "CREATED_DATE",
                    "HEIGHT",
                    "WEIGHT",
                    "BMI",
                    "BP",
                    "DIA_BP",
                    "FS",
                    "PP",
                    "PULSE",
                    "A1C",
                    "COMPLAINTS",
                    "PATIENT_TYPE",
                    "DIAGNOSIS",
                    "NOTES",
                ]
            ]
            .dropna(subset=["NFID"])
            .drop_duplicates(subset=["APPOINT_ID"])
            .reset_index(drop=True)
        )
        # Convert all number based cells to numeric and coerce errors to accumulate the NaNs
        numeric_cols = [
            "APPOINT_ID",
            "NFID",
            "HEIGHT",
            "WEIGHT",
            "BMI",
            "BP",
            "DIA_BP",
            "FS",
            "PP",
            "PULSE",
            "A1C",
        ]

        self.meas_df = pd.concat(
            [
                self.meas_df.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce"),
                self.meas_df.loc[:, self.meas_df.columns.difference(numeric_cols)],
            ],
            axis=1,
        ).reset_index(drop=True)

        self.meas_df["NFID"] = self.meas_df["NFID"].astype(int)

        # Nan strings to Nan in Patient type:
        self.meas_df["PATIENT_TYPE"] = (
            self.meas_df["PATIENT_TYPE"]
            .replace("Nan", np.nan, regex=True)
            .str.replace("\d+", " ", regex=True)
            .str.replace("\W+", " ", regex=True)
        )
        self.meas_df["DIAGNOSIS"] = self.meas_df["DIAGNOSIS"].replace(
            "Nan", np.nan, regex=True
        )
        self.meas_df["COMPLAINTS"] = self.meas_df["COMPLAINTS"].replace(
            "Nan", np.nan, regex=True
        )
        self.meas_df["PATIENT_TYPE"] = self.meas_df["PATIENT_TYPE"].replace(
            "NON DM", "NON-DM", regex=True
        )
        self.meas_df["COMPLAINTS"] = self.meas_df["COMPLAINTS"].replace(
            "NON MS", "NON-MS", regex=True
        )

        # replace 0 with Nan for every column except APPOINT_ID:
        self.meas_df.loc[:, self.meas_df.columns != "APPOINT_ID"] = self.meas_df.loc[
            :, self.meas_df.columns != "APPOINT_ID"
        ].replace(0, np.nan)

        # outliers removal:
        # https://stackoverflow.com/questions/35827863/remove-outliers-in-pandas-dataframe-using-percentiles
        outlier_cols = [
            "HEIGHT",
            "WEIGHT",
            "BMI",
            "BP",
            "DIA_BP",
            "FS",
            "PP",
            "PULSE",
            "A1C",
        ]

        filt_df = self.meas_df[outlier_cols]
        low = 0.01
        high = 0.99
        quant_df = filt_df.quantile([low, high])

        filt_df = filt_df.apply(
            lambda x: x[
                (x > quant_df.loc[low, x.name]) & (x < quant_df.loc[high, x.name])
            ],
            axis=0,
        )
        self.meas_df[outlier_cols] = filt_df

        # extract T2DM new, DM new, DM recent
        self.meas_df["NEW_T2DM"] = (
            self.meas_df["NOTES"].str.contains("new|recent", regex=True).fillna(False)
        )

        return self.meas_df

    def map_complications(self):
        # map generic names to drug class combinations:
        # TODO: exclude no angina (check for negations)
        complication_types = dict_partial_match(COMPLICATION_TYPES)
        self.meas_df["COMPLICATIONS"] = (
            self.meas_df["COMPLAINTS"]
            + self.meas_df["NOTES"]
            + self.meas_df["DIAGNOSIS"]
        ).apply(lambda x: complication_types[x])
        return self.meas_df

    def map_patient_types(self):
        # map patient types:
        # TODO: check for types in either DIAGNOSIS or NOTES
        patient_types = dict_partial_match(PATIENT_TYPES)
        self.meas_df["PATIENT_TYPE"] = self.meas_df["PATIENT_TYPE"].apply(
            lambda x: patient_types[x]
        )
        return self.meas_df

    def measurement_aggregator(self, group):
        output_dict = {}

        def min_two_values(series):
            if len(series) < 2:
                return np.nan
            else:
                return series.tolist()

        # columns where average value should be taken
        mean_cols = ["HEIGHT", "WEIGHT", "BMI", "BP", "DIA_BP", "PULSE"]
        output_dict = {col: group[col].mean() for col in mean_cols}

        # patients with min two A1C values are considered
        output_dict["A1C"] = min_two_values(group["A1C"])
        output_dict["A1C_counts"] = group["A1C"].dropna().count()
        output_dict["A1C_first_two_values"] = group["A1C"].dropna().tolist()[:2]

        # FF, PP and CREATED_DATE are converted to lists for plotting
        output_dict["FS"] = group["FS"].tolist()
        output_dict["PP"] = group["PP"].tolist()
        output_dict["CREATED_DATE"] = group["CREATED_DATE"].tolist()

        # COMPLAINTS are aggregated as strings
        # DIAGNOSIS, NOTES and PATIENT_TYPE are converted to lists for further processing
        output_dict["COMPLAINTS"] = group["COMPLAINTS"].dropna().str.cat(sep=". ")
        output_dict["DIAGNOSIS"] = group["DIAGNOSIS"].dropna().str.cat(sep=". ")
        output_dict["NOTES"] = group["NOTES"].dropna().str.cat(sep=". ")

        # combine existing sets of COMPLICATIONS, PATIENT_TYPE into a single set:
        # PATIENT_TYPE is one of DM, IDDM, THY, etc. and a patient can have multiple types:
        output_dict["COMPLICATIONS"] = set().union(
            *group["COMPLICATIONS"].dropna().tolist()
        )
        output_dict["PATIENT_TYPE"] = set().union(
            *group["PATIENT_TYPE"].dropna().tolist()
        )

        output_dict["NEW_T2DM"] = group["NEW_T2DM"].any()

        # extract duration of diabetes from NOTES; eg.'DM since 2001'
        def convert_to_timestamp(x):
            # convert str like '2001' to days of diabetes:
            try:
                return (pd.Timestamp.today() - pd.Timestamp(x)).days
            except:
                return np.nan

        dm_since = set(
            group["NOTES"]
            .dropna()
            .str
            # findall instances of years (4-digits) in the str:
            .findall("[0-9]{4}")
            # handle multiple years in the same str:
            .apply(lambda x: x[0] if x else np.nan)
            .dropna()
            .tolist()
        )
        dm_since = dm_since.pop() if dm_since else np.nan
        output_dict["DURATION"] = convert_to_timestamp(dm_since)

        return pd.Series(output_dict)

    def combine_patient_details(self):
        self.meas_pat_df = (
            self.meas_df.groupby(["NFID"]).apply(self.measurement_aggregator).dropna()
        )
        # check if patient is in NFID of meas details:
        self.pat_df = self.pat_df.loc[
            self.pat_df["PATIENT_NFID"].isin(self.meas_pat_df.index.values)
        ].reset_index(drop=True)
        self.pat_df = self.pat_df.set_index("PATIENT_NFID")
        self.pat_df.index.names = ["NFID"]

        self.pat_df["PATIENT_GENDER"] = (
            self.pat_df["PATIENT_GENDER"].str.lower().astype("category").cat.codes
        )

        # from sklearn.preprocessing import OrdinalEncoder
        # ord_enc = OrdinalEncoder()
        # self.pat_df['PATIENT_GENDER'] = ord_enc.fit_transform(self.pat_df[['PATIENT_GENDER']]).astype(int)

        # get the duration of diabetes:
        self.pat_df["DURATION"] = (
            pd.Timestamp.today() - self.pat_df["created_time"]
        ).dt.days

        self.meas_pat_df["AGE"] = self.pat_df["PATIENT_AGE"]
        self.meas_pat_df["GENDER"] = self.pat_df["PATIENT_GENDER"]

        # fillna with created date duration values when 'DM since' was not available:
        self.meas_pat_df["DURATION"] = self.meas_pat_df["DURATION"].fillna(
            self.pat_df["DURATION"]
        )

        return self.meas_pat_df
