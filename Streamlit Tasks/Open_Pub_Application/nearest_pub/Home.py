import streamlit as st
import pandas as pd
import numpy as np

################################## DATAFRAME CODES #######################################
df = pd.read_csv('data/open_pubs.csv')
df.columns = ['fsa_id', 'name', 'address', 'postcode', 'easting', 'northing', 'latitude', 'longitude', 'local_authority']
df = df.replace('\\N', np.NaN)
df = df.dropna()
df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype('float64')
df.drop(df[['easting', 'northing']], axis = 1, inplace = True)
col = ['fsa_id', 'name', 'address', 'postcode', 'easting', 'northing', 'latitude', 'longitude', 'local_authority']

################################## VARIABLE CODES #######################################
total_pubs = "Total Pubs in United Kingdoms: \n" + str(len(df['fsa_id'].unique()))
total_local_authority = "Total Local Authorities in United Kingdoms: \n" + str(len(df['local_authority'].unique()))
link = "Here is the data for pub: \n" + str("https://drive.google.com/file/d/16SpndGibG0MKTWS5KN8HFqGs-e7LvZTG/view")
columns = "Columns in the Data Represents: \n" + str(col)

################################## STREAMLIT CODES #######################################
st.set_page_config(page_title="Hello", page_icon="👋")
st.title(" :champagne: *'Choose the Booze!'* :beer: ")
st.header("*'We are grapeful to have you on our app!'*:wine_glass:")
st.header(" ")
st.subheader(link)
st.subheader(columns)
st.subheader(total_pubs)
st.subheader(total_local_authority)

################################## BACKGROUND CODES #######################################
page_bg_img = '''
<style>
.stApp {
background-image: url("https://i.pinimg.com/originals/66/77/0c/66770cd3a86faccd6537a77ae34e2067.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)