import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# SETTING PAGE CONFIGURATION
st.set_page_config(layout="wide", page_title="Bay Wheels Timelapse", page_icon=":bike:")

# LOAD DATA
@st.cache_data
def load_data():
    data = pd.read_csv("bay_wheels_october.csv")  # Update this to your file path
    data["started_at"] = pd.to_datetime(data["started_at"])  # Ensure timestamps are datetime
    data["hour"] = data["started_at"].dt.hour  # Extract the hour from timestamps
    data["day_of_week"] = data["started_at"].dt.day_name()  # Extract day of week
    return data

data = load_data()

# AGGREGATE DATA BY HOUR AND DAY
@st.cache_data
def aggregate_data(data):
    agg_data = (
        data.groupby(["start_lat", "start_lng", "hour", "day_of_week"])
        .size()
        .reset_index(name="trip_count")
    )
    return agg_data

agg_data = aggregate_data(data)

# CALCULATE MIDPOINT FOR INITIAL MAP VIEW
@st.cache_data
def calculate_midpoint(latitudes, longitudes):
    return np.average(latitudes), np.average(longitudes)

midpoint = calculate_midpoint(agg_data["start_lat"], agg_data["start_lng"])

# USER CONTROLS
st.title("Bay Wheels Timelapse with Filters")
st.write("Watch the movement of trips across the Bay Area throughout the day with a 3D timelapse and advanced filtering.")

# Day of Week Filter
day_of_week = st.selectbox("Select Day of the Week", options=["All"] + list(data["day_of_week"].unique()))

# Hour Slider
hour = st.slider("Select Hour", 0, 23, step=1, value=0)

# Granularity Controls
radius = st.slider("Hexagon Radius (meters)", 100, 1000, step=100, value=200)
elevation_scale = st.slider("Elevation Scale", 1, 10, step=1, value=4)

# Play/Pause Timelapse Buttons
play = st.button("Play Timelapse")
pause = st.button("Pause Timelapse")

# Filter Data
if day_of_week != "All":
    filtered_data = agg_data[(agg_data["hour"] == hour) & (agg_data["day_of_week"] == day_of_week)]
else:
    filtered_data = agg_data[agg_data["hour"] == hour]

# 3D Hexagon Map Layer
layer = pdk.Layer(
    "HexagonLayer",
    data=filtered_data,
    get_position=["start_lng", "start_lat"],
    radius=radius,
    elevation_scale=elevation_scale,
    elevation_range=[0, 1000],
    extruded=True,
    pickable=True,
)

# Map View State
view_state = pdk.ViewState(
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=11,
    pitch=50,
)

# Pydeck Map
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Trips: {elevationValue}"},
    map_style="mapbox://styles/mapbox/light-v9",
)

# Display the map
map_placeholder = st.empty()
map_placeholder.pydeck_chart(deck)

# Timelapse Logic
if play:
    for h in range(hour, 24):
        if pause:  # Stop timelapse if Pause button is clicked
            break

        if day_of_week != "All":
            filtered_data = agg_data[(agg_data["hour"] == h) & (agg_data["day_of_week"] == day_of_week)]
        else:
            filtered_data = agg_data[agg_data["hour"] == h]

        layer = pdk.Layer(
            "HexagonLayer",
            data=filtered_data,
            get_position=["start_lng", "start_lat"],
            radius=radius,
            elevation_scale=elevation_scale,
            elevation_range=[0, 1000],
            extruded=True,
            pickable=True,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Trips: {elevationValue}"},
            map_style="mapbox://styles/mapbox/light-v9",
        )

        # Update map in placeholder
        map_placeholder.pydeck_chart(deck)

        # Progress through time
        st.write(f"Displaying trips for {h}:00 on {day_of_week if day_of_week != 'All' else 'All Days'}")
        time.sleep(1)  # Pause for 1 second per frame
