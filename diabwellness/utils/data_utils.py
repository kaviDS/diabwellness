# Copyright 2022 Diabwellness.ai, Inc.
# All rights reserved

"""Utility function for data processing."""

import re
import numpy as np
import pandas as pd
from scipy import stats


class dict_partial_match_list(dict):
    def __getitem__(self, x):
        result_list = []
        for k in self.keys():
            if str(k).lower() in str(x).lower() or str(x).lower() in str(k).lower():
                result_list.append(self.get(k))
        if result_list:
            return result_list
        else:
            return np.nan


def map_generic_names(pres_df, drug_df):
    # extract relevant columns:
    pres_df = pres_df[
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
    pres_df = (
        pres_df[
            ~(
                (
                    pres_df["DRUG_NAME"].str.contains(
                        "lantus|tresiba|basalog", case=False, regex=True
                    )
                )
                & (pres_df["QTY"] == 0)
            )
        ]
        .dropna()
        .reset_index(drop=True)
    )
    # pres_df = pres_df.astype({"PATIENT_ID": 'int'}, errors = 'ignore')
    pres_df["PATIENT_ID"] = pres_df["PATIENT_ID"].astype(int)
    # extract relevant columns:
    drug_df = (
        drug_df[["DRUG_ID", "DRUG_NAME", "GENERIC_NAME"]]
        .dropna()
        .reset_index(drop=True)
    )

    # map drug names to generic names based on drug id:
    # drug_df['DRUG_ID'] and pres_df['ITEM_ID'] are the same:
    drug_master = drug_df.set_index("DRUG_ID")["GENERIC_NAME"]
    pres_df["GENERIC_NAME"] = pres_df["ITEM_ID"].replace(drug_master)

    return pres_df


DRUG_CLASSES = {
    (
        "Glimepiride",
        "Glimepride",
        "Gliclazide",
        "Glipizide",
        "Glibenclamide",
    ): "Sulfonylureas",
    ("Metformin",): "Biguanide",
    ("Piogoitazone",): "PPAR gamma agonist (Glitazone)",
    (
        "Sitagliptin",
        "Vildagliptin",
        "Saxagliptin",
        "Linagliptin",
        "Teneligliptin",
    ): "DPP4 Inhibitors (Gliptins)",
    (
        "Dapagliflozin",
        "Empagliflozin",
        "Canagliflozin",
    ): "SGLT2- Inhibitors (Gliflozins)",
    (
        "Acarbose",
        "Voglibose",
    ): "Alpha Glucosidase inhibitors",
    ("Repaglinide",): "Glinides",
    ("Bromocriptine",): "Dopamine agonist",
    (
        "Hydroxychloroquine",
        "Hydroxychloroquin",
    ): "Anti Inflammator",
    ("Saroglitazar",): "PPAR gamma & alpha agonist",
    ("Insulin",): "Insulin",
}


def map_drug_classes(pres_df):
    # map generic names to drug class combinations:
    drug_classes = dict_partial_match(DRUG_CLASSES)
    pres_df["DRUG_CLASS"] = pres_df["GENERIC_NAME"].apply(lambda x: drug_classes[x])
    return pres_df.dropna(subset=["DRUG_CLASS"]).reset_index(drop=True)


def extract_drug_combinations(pres_df):
    # collect all anti-diabetic drugs across multiple tablets in the first appointment:
    def combined_list(input_list):
        # return tuple(sorted(set([a for b in input_list for a in b])))
        return tuple(sorted(set().union(*input_list)))

    pres_df = (
        pres_df.groupby("PATIENT_ID").apply(
            lambda x: (
                x.groupby(["APPOINT_ID"]).apply(
                    lambda y: combined_list(y["DRUG_CLASS"].to_list())
                )
            ).iloc[0]
        )
    ).to_frame(name="DRUG_COMBINATION")

    # number of drugs in each combination:
    pres_df["DRUG_COUNT"] = pres_df["DRUG_COMBINATION"].apply(lambda x: len(x))
    pres_df.index.name = "NFID"
    return pres_df


def preprocess_measurements(meas_df):
    # preprocess the measurement details:
    meas_df = (
        meas_df[
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

    meas_df = pd.concat(
        [
            meas_df.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce"),
            meas_df.loc[:, meas_df.columns.difference(numeric_cols)],
        ],
        axis=1,
    ).reset_index(drop=True)

    meas_df["NFID"] = meas_df["NFID"].astype(int)

    # Nan strings to Nan in Patient type:
    meas_df["PATIENT_TYPE"] = (
        meas_df["PATIENT_TYPE"]
        .replace("Nan", np.nan, regex=True)
        .str.replace("\d+", " ", regex=True)
        .str.replace("\W+", " ", regex=True)
    )
    meas_df["DIAGNOSIS"] = meas_df["DIAGNOSIS"].replace("Nan", np.nan, regex=True)
    meas_df["COMPLAINTS"] = meas_df["COMPLAINTS"].replace("Nan", np.nan, regex=True)
    meas_df["PATIENT_TYPE"] = meas_df["PATIENT_TYPE"].replace(
        "NON DM", "NON-DM", regex=True
    )
    meas_df["COMPLAINTS"] = meas_df["COMPLAINTS"].replace(
        "NON MS", "NON-MS", regex=True
    )

    # replace 0 with Nan for every column except APPOINT_ID:
    meas_df.loc[:, meas_df.columns != "APPOINT_ID"] = meas_df.loc[
        :, meas_df.columns != "APPOINT_ID"
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

    filt_df = meas_df[outlier_cols]
    low = 0.01
    high = 0.99
    quant_df = filt_df.quantile([low, high])

    filt_df = filt_df.apply(
        lambda x: x[(x > quant_df.loc[low, x.name]) & (x < quant_df.loc[high, x.name])],
        axis=0,
    )
    meas_df[outlier_cols] = filt_df

    # extract T2DM new, DM new, DM recent
    meas_df["NEW_T2DM"] = (
        meas_df["NOTES"].str.contains("new|recent", regex=True).fillna(False)
    )

    return meas_df


def a1c_aggregator(group):
    output_dict = {}

    def first_two_values(series):
        if len(series) < 2:
            return [np.nan, np.nan]
        else:
            return series.iloc[:2].tolist()

    # patients with min two A1C values are considered
    a1c_values = first_two_values(group["A1C"].dropna())
    output_dict["A1C_1"], output_dict["A1C_2"] = a1c_values
    output_dict["NEW_T2DM"] = group["NEW_T2DM"].any()

    return pd.Series(output_dict)


def measurement_aggregator(group):
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
    output_dict["PATIENT_TYPE"] = set().union(*group["PATIENT_TYPE"].dropna().tolist())

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


def calculate_efficacy(pres_filt_df, meas_filt_df):
    # combine the drug combinations and A1C values, count the drugs:
    efficacy_df = pd.concat([pres_filt_df, meas_filt_df], axis=1).dropna()
    efficacy_df = efficacy_df.groupby("DRUG_COMBINATION").apply(
        lambda x: [
            x["A1C_1"].count(),
            (x["NEW_T2DM"] == True).sum(),
            x["A1C_1"].mean(),
            x["A1C_2"].mean(),
            (x["A1C_2"] - x["A1C_1"]).mean(),
        ]
        + list(stats.ttest_ind(a=x.A1C_1, b=x.A1C_2, equal_var=False))
    )
    efficacy_df = pd.DataFrame(
        efficacy_df.tolist(),
        index=efficacy_df.index,
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
    return efficacy_df


def extract_biguanide_combinations(efficacy_df, drug_counts=2):
    efficacy_df = efficacy_df[
        (efficacy_df.index.map(len) == drug_counts)
        & (
            efficacy_df.index.to_series().apply(
                lambda x: True if "Biguanide" in x else False
            )
        )
    ]
    # sort according to p-values:
    counts_cond = efficacy_df["patient_counts"] > 0
    return efficacy_df.loc[counts_cond].sort_values(["p_value"], ascending=True)


def combine_patient_details(meas_filt_df, pat_df):
    # check if patient is in NFID of meas details:
    pat_df = pat_df.loc[
        pat_df["PATIENT_NFID"].isin(meas_filt_df.index.values)
    ].reset_index(drop=True)
    pat_df = pat_df.set_index("PATIENT_NFID")
    pat_df.index.names = ["NFID"]

    pat_df["PATIENT_GENDER"] = (
        pat_df["PATIENT_GENDER"].str.lower().astype("category").cat.codes
    )

    # from sklearn.preprocessing import OrdinalEncoder
    # ord_enc = OrdinalEncoder()
    # pat_df['PATIENT_GENDER'] = ord_enc.fit_transform(pat_df[['PATIENT_GENDER']]).astype(int)

    # get the duration of diabetes:
    pat_df["DURATION"] = (pd.Timestamp.today() - pat_df["created_time"]).dt.days

    meas_filt_df["AGE"] = pat_df["PATIENT_AGE"]
    meas_filt_df["GENDER"] = pat_df["PATIENT_GENDER"]

    # fillna with created date duration values when 'DM since' was not available:
    meas_filt_df["DURATION"] = meas_filt_df["DURATION"].fillna(pat_df["DURATION"])

    return meas_filt_df


class dict_partial_match(dict):
    def __getitem__(self, x):
        if x == np.nan:
            return np.nan
        result_set = set()
        for k in self.keys():
            for t in k:
                # exact match with word boundaries excluding hyphen:
                # https://stackoverflow.com/questions/39684942/how-to-make-word-boundary-b-not-match-on-dashes
                if re.search(rf"(?<!-)\b({t})\b", str(x), re.IGNORECASE):
                    result_set.add(self.get(k))
        if bool(result_set):
            return result_set
        else:
            return np.nan


PATIENT_TYPES = {
    ("DM",): "DM",
    ("NON-DM",): "NON-DM",
    ("NON-MS",): "NON-MS",
    (
        "THY",
        "Hypothyroid",
        "Hypothyroidism",
        "Thyroid",
    ): "THY",
    ("IDDM",): "IDDM",
    ("GDM",): "GDM",
    (
        "InflDisorder",
        "Inf",
    ): "InflDisorder",
    (
        "HT",
        "HTN",
    ): "HT",
    (
        "Hlip",
        "Hyperlipidemia",
    ): "Hlip"
    # ("Allergy",): "Allergy",
    # ("Anemia",): "Anemia",
    # ("Cancer",): "Cancer",
}

COMPLICATION_TYPES = {
    (
        "CAD",
        "Angina",
        "Unstable angina",
        "ASMI",
        "AWMI",
        "IWMI",
        "PTCA",
        "CABG",
        "heart attack",
        "infarct",
        "ischemia",
    ): "CAD",
    (
        "Stroke",
        "hemiplegia",
        "hemiparesis",
        "cerebral infarct",
        "VBI",
        "vertebrobasilar infarct",
        "pontine infarct",
        "PICA",
        "AICA",
        "cerebellar infarct",
    ): "CVD",
    (
        "Intermittent claudication",
        "pain legs on walking",
        "pain calf muscles on walking",
    ): "PVD",
    (
        "PN",
        "painful peripheral Neuropathy",
        "numbness",
        "tingling",
        "burning feet",
        "numb feet",
        "numb limbs",
        "DPN",
    ): "DPN",
    (
        "DKD",
        "CKD",
        "Microalbuminuria",
        "Macroalbuminuria",
        "DN",
        "Nephropathy",
        "diabetic nephropathy",
    ): "DN",
    (
        "DR",
        "PDR",
        "NPDR",
        "Diabetic retinopathy",
    ): "DR",
    (
        "Foot ulcer",
        "bleb",
        "wound",
        "amputee",
        "fissure",
        "callus",
        "corn",
        "DFU",
        "Nonhealing ulcer",
    ): "DFU",
}


def map_complications(meas_df):
    # map generic names to drug class combinations:
    # TODO: exclude no angina (check for negations)
    complication_types = dict_partial_match(COMPLICATION_TYPES)
    meas_df["COMPLICATIONS"] = (
        meas_df["COMPLAINTS"] + meas_df["NOTES"] + meas_df["DIAGNOSIS"]
    ).apply(lambda x: complication_types[x])
    return meas_df


def map_patient_types(meas_df):
    # map patient types:
    # TODO: check for types in either DIAGNOSIS or NOTES
    patient_types = dict_partial_match(PATIENT_TYPES)
    meas_df["PATIENT_TYPE"] = meas_df["PATIENT_TYPE"].apply(lambda x: patient_types[x])
    return meas_df
