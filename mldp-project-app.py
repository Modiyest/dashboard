import streamlit as st
import pandas as pd 
import joblib


st.title("London Property Price Predictor")
st.write("""
# 
Simply fill in the details on the left, and the app will provide an estimated price for your desired property.
""")

st.sidebar.header('User Input Parameters')
st.sidebar.write("Use the controls to configure the property details.")

model = joblib.load('London Property Listings Dataset.pkl')
areas_avg_price = {
    "Area_Bromley": 418750.0, "Area_Croydon": 435294.1176470588, "Area_Eastern": 1001684.3915590268, "Area_Eastern Central": 1410220.4217687075, 
    "Area_Enfield": 501597.28712871287, "Area_Harrow": 510628.26086956525, "Area_Ilford": 585666.6666666666, "Area_Kingston": 570000.0, 
    "Area_Kingston upon Thames": 800996.6666666666, "Area_North Western": 1237283.4899466557, "Area_Northern": 831295.2083578575, 
    "Area_South Eastern": 692104.7799433026, "Area_South Western": 1516724.372564152, "Area_Sutton": 661666.6666666666, "Area_Twickenham": 851258.7475345167, 
    "Area_Western Central": 1625819.108695652, "Area_Western and Paddington": 1706839.3389084507
}
property_type = [
    "Apartment", "Flat", "House",
    "Semi-Detached", "Terraced"
]


def user_input_features():
    areas = list(areas_avg_price.keys()) 
    st.sidebar.subheader("Property Details")
    size = st.sidebar.slider('Size', 52.0, 1500000.0, 52.0)
    bedroom = st.sidebar.slider('Bedrooms', 1.0, 10.0, 1.0, step=1.0)
    bathroom = st.sidebar.slider('Bathrooms', 1.0, 144.0, 1.0, step=1.0)
    st.sidebar.subheader("Area & Property Type")
    selected_area = st.sidebar.selectbox('Select an Area:', areas)
    area_ohe = {f"Area_{area}": (1 if area == selected_area else 0) for area in areas}
    selected_property = st.sidebar.selectbox('Select Property Type', property_type)
    property_type_ohe = {f"Property Type_{ptype}": (1 if ptype == selected_property else 0) for ptype in property_type}
    Area_Avg_Price = areas_avg_price.get(selected_area, 0)
    data = {'Bathrooms': bathroom,
            'Bedrooms': bedroom,
            'Size': size,
            'Area_Avg_Price': Area_Avg_Price,
            **property_type_ohe,
            **area_ohe}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("Average Property Prices by Area")
area_prices_df = pd.DataFrame(list(areas_avg_price.items()), columns=["Area", "Average Price"])
area_prices_df.set_index("Area", inplace=True)
st.bar_chart(area_prices_df)

prediction = model.predict(df)
st.subheader('Predicted Property Price')
st.success(f"£ {prediction[0]:,.2f}")
