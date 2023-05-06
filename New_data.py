#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


meas_filt_dfc = pd.read_csv("meas_filt_dfc.tsv", index_col=0, sep='\t')
import json
def load_list(x):
    x = re.sub(r'\bnan\b', 'NaN', x)
    return json.loads(x)
meas_filt_dfc = meas_filt_dfc.applymap(load_list)
meas_filt_dfc.head()


# In[3]:


meas_filt_dfc = meas_filt_dfc[['AGE','BP','DIA_BP','DM_SINCE','BMI','PATIENT_TYPE_HT','PATIENT_TYPE_Hlip',"COMPLICATION_CAD","COMPLICATION_CVD","COMPLICATION_DFU","COMPLICATION_DN","COMPLICATION_DPN","COMPLICATION_DR","COMPLICATION_PVD"]]
pres_filt_df = pd.read_csv("pres_filt_df.tsv", index_col=0, sep='\t')


# In[4]:


pres_filt_df.info()


# In[5]:


meas_filt_dfc.info()


# In[9]:


def create_target_df(input_df,pres_filt_df):
    output_df = pd.DataFrame()
    for col in input_df.columns:
        if col == "BP" or col == "DIA_BP" or col == "BMI":
            output_df[col] = input_df[col].apply(lambda x: np.mean(x))
        else:
            output_df[col] = input_df[col].apply(lambda x: x[0])
        
    output_df["MICRO"] = output_df.apply(lambda x: x['COMPLICATION_CVD'] or x['COMPLICATION_CAD'] or x['COMPLICATION_PVD'], axis=1 )
    output_df["MACRO"] = output_df.apply(lambda x: x['COMPLICATION_DFU'] or x['COMPLICATION_DN'] or x['COMPLICATION_DPN'] or x['COMPLICATION_DR'], axis=1 )
    output_df['FINAL_HT'] = output_df.apply(lambda x: x['BP'] >= 140 and x['DIA_BP'] >= 90 ,axis=1 ).map({False: 0,True: 1}).astype(int)
    output_df = pd.concat([output_df,pres_filt_df], axis=1).dropna()
    output_df["DRUG"] = output_df['DRUG_COMBINATION'].apply(lambda x : all(i in x for i in ('Insulin', 'Sulfonylureas'))).map({False: 0,True: 1}).astype(int)
    output_df = output_df.drop(['COMPLICATION_CAD','COMPLICATION_CVD','COMPLICATION_DFU','COMPLICATION_DN','COMPLICATION_DPN','COMPLICATION_DR','COMPLICATION_PVD','DRUG_COUNT'], axis=1)
    return output_df 
    


# In[10]:


create_target_df(meas_filt_dfc,pres_filt_df)


# In[ ]:




