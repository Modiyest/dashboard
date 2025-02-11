import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load('London_Property_Listings_Dataset.pkl')

# Streamlit App Title
st.title("London Property Price Predictor")
st.write("""
# 
Simply fill in the details on the left, and the app will provide an estimated price for your desired property.
""")

# Sidebar for User Input
st.sidebar.header('User Input Parameters')
st.sidebar.write("Use the controls to configure the property details.")

# Area Average Prices Dictionary
areas_avg_price = {
    "Area_Bromley": 418750.0, "Area_Croydon": 435294.1176470588, "Area_Eastern": 1001684.3915590268, "Area_Eastern Central": 1410220.4217687075, 
    "Area_Enfield": 501597.28712871287, "Area_Harrow": 510628.26086956525, "Area_Ilford": 585666.6666666666, "Area_Kingston": 570000.0, 
    "Area_Kingston upon Thames": 800996.6666666666, "Area_North Western": 1237283.4899466557, "Area_Northern": 831295.2083578575, 
    "Area_South Eastern": 692104.7799433026, "Area_South Western": 1516724.372564152, "Area_Sutton": 661666.6666666666, "Area_Twickenham": 851258.7475345167, 
    "Area_Western Central": 1625819.108695652, "Area_Western and Paddington": 1706839.3389084507
}

# Property Types
property_types = ["Apartment", "Flat", "House", "Semi-Detached", "Terraced"]

# Function to Collect User Input
def user_input_features():
    areas = list(areas_avg_price.keys())

    st.sidebar.subheader("Property Details")
    size = float(st.sidebar.slider('Size (sq ft)', 52.0, 1500000.0, 52.0))
    bedrooms = int(st.sidebar.slider('Bedrooms', 1, 10, 1))
    bathrooms = int(st.sidebar.slider('Bathrooms', 1, 144, 1))

    st.sidebar.subheader("Area & Property Type")
    selected_area = st.sidebar.selectbox('Select an Area:', areas)
    selected_property = st.sidebar.selectbox('Select Property Type', property_types)

    # One-hot encoding for Area
    area_ohe = {f"Area_{area}": int(area == selected_area) for area in areas}

    # One-hot encoding for Property Type
    property_type_ohe = {f"Property Type_{ptype}": int(ptype == selected_property) for ptype in property_types}

    # Get Average Price for Area
    area_avg_price = areas_avg_price.get(selected_area, 0.0)

    # Combine all inputs into a dictionary
    data = {
        'Size': size,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Area_Avg_Price': area_avg_price,
        **area_ohe,
        **property_type_ohe
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    return df

# Get User Inputs
df = user_input_features()

# Show Bar Chart of Average Prices by Area
st.subheader("Average Property Prices by Area")
area_prices_df = pd.DataFrame(list(areas_avg_price.items()), columns=["Area", "Average Price"])
area_prices_df.set_index("Area", inplace=True)
st.bar_chart(area_prices_df)

# Ensure Model and Input Data Match
try:
    # Ensure all columns in the trained model are in df
    model_columns = joblib.load('London Property Listings Dataset.pkl').feature_names_in_
    
    # Add missing columns with zero values
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match model training order
    df = df[model_columns]

    # Predict Property Price
    prediction = model.predict(df)
    st.subheader('Predicted Property Price')
    st.success(f"£ {prediction[0]:,.2f}")

except Exception as e:
    st.error(f"Error in prediction: {e}")
