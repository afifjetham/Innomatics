import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk

################################## DATAFRAME CODES #######################################
df = pd.read_csv('data/open_pubs.csv')
df.columns = ['fsa_id', 'name', 'address', 'postcode', 'easting', 'northing', 'latitude', 'longitude', 'local_authority']
df = df.replace('\\N', np.NaN)
df = df.dropna()
df[['latitude', 'longitude']] = df[['latitude', 'longitude']].astype('float64')
df.drop(df[['easting', 'northing']], axis = 1, inplace = True)
df_geoloc = df[['latitude', 'longitude']]

################################## STREAMLIT CODES #######################################
st.set_page_config(page_title="Pub Location", page_icon="📍")
st.header("*Find the Nearest Pub* :beer: ")


################################## EUCLIDEAN DISTANCE CODES #######################################
 
def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** ( 1 / 2)
 
# Function to calculate K closest points
def kClosest(points, target, K):
    pts = []
    n = len(points)
    d = []
 
    for i in range(n):
        d.append({
            "first": distance(points[i][0], points[i][1], target[0], target[1]),
            "second": i
        })
    

    d = sorted(d, key=lambda l:l["first"])
    
    for i in range(K):
        pt = []
        pt.append(points[d[i]["second"]][0])
        pt.append(points[d[i]["second"]][1])
        pts.append(pt)
   
# Calling DataFrame constructor on list
    df_nearest_loc = pd.DataFrame(pts,columns=['latitude','longitude'])
    st.map(df_nearest_loc)
    
# Driver code
df_geoloc=df[['latitude','longitude']]
points = df_geoloc.values.tolist()

lat = st.number_input('Enter Latitude',format="%.5f")
log = st.number_input('Enter Longitude',format="%.6f",key=int)

target = [lat,log]
K = 5

kClosest(points, target, K)

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