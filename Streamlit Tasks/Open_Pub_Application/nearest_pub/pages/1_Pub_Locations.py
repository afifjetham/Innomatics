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
df_geoloc = df[['latitude', 'longitude']]

################################## STREAMLIT CODES #######################################
st.set_page_config(page_title="Pub Location", page_icon="🗺️")
st.header(" *Locations of all the Pubs in UK* :beer: ")

authority = st.selectbox('Select a Local Authority', list(df['local_authority'].unique()))


button_click= st.button('Search')
if button_click==True: 
    auth= df[df['local_authority']==authority]
    count=len(auth)
    st.write("Number of Pubs in the area:",count)
    auth=auth[['latitude','longitude']]
    st.map(auth)


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