import pickle
import json
import numpy as np
import pandas as pd
import os
from scipy.special import boxcox, inv_boxcox

__locations = None
__data_columns = None
__model_buy = None
__model_rent = None
__location_indices = None
__scaler = None


def get_buy_estimate(city, location, sqft, bhk, bath):
    other_features_n = 17
    lmbda = 0.09833917166260583

    try:
        loc_index = __location_indices.index(location) + other_features_n
    except:
        loc_index = -1

    scaler_default = {'Lift': 1.781519, 'BHK_imp': bhk, 'Bedrooms_imp': 1.753627, 'Bathrooms_imp': bath,
                      'No_of_Towers': 3.522327, 'SuperArea(insqft)_imp': 9.031997,
                      'PlotArea(insqft)_imp': boxcox(sqft, lmbda), 'CarpetArea_imp': 6.384659, 'FAR': 8.551286}

    scaler_df = pd.DataFrame(scaler_default, index=[0])

    vals = __scaler.transform(scaler_df)[0]
    scalers = {key: vals[i] for i, key in enumerate(scaler_default)}

    default_data = {
        'BHK_imp': 0.240513,
        'Bathrooms_imp': 0.116393,
        'No_of_Towers': 0.297662,
        'PlotArea(insqft)_imp': 0.480518,
        'CarpetArea_imp': 0.511952,
        'FAR': 0.031309,
        'Direction Facing': 5.129946,
        'Property Ownership': 0.589478,
        'FlooringRating': 0.266187,
        'Total_No_of_Floors': 1.172212,
        'Floor_Category': 0.821942,
        'Property_Age_Enc': 3.677158,
        'Furnishing_Enc': 1.445594,
        'Loading_Factor_Status': 1.544964,
        'Swimming Pool_Y': 0.297662,
        'is_Spacious_Spacious': 0.995504,
        'City_Mumbai': 0.0,
    }

    if city == 'Mumbai':
        default_data['City_Mumbai'] = 1.0

    data = [scalers[key] if key in scalers else default_data[key] for key in default_data]
    x = np.concatenate((np.array(data), np.zeros(187)))

    if loc_index >= 0:
        x[loc_index + other_features_n] = 1

    return round(np.expm1(__model_buy.predict([x])[0]) / 100000, 2)


def get_buy_estimate(city, location, sqft, bhk, bath):
    other_features_n = 17
    lmbda = 0.09833917166260583

    try:
        loc_index = __location_indices["buy"].index(location) + other_features_n
    except:
        loc_index = -1

    scaler_default = {'Lift': 1.781519, 'BHK_imp': bhk, 'Bedrooms_imp': 1.753627, 'Bathrooms_imp': bath,
                      'No_of_Towers': 3.522327, 'SuperArea(insqft)_imp': 9.031997,
                      'PlotArea(insqft)_imp': boxcox(sqft, lmbda), 'CarpetArea_imp': 6.384659, 'FAR': 8.551286}

    scaler_df = pd.DataFrame(scaler_default, index=[0])

    vals = __scaler["buy"].transform(scaler_df)[0]
    scalers = {key: vals[i] for i, key in enumerate(scaler_default)}

    default_data = {
        'BHK_imp': 0.240513,
        'Bathrooms_imp': 0.116393,
        'No_of_Towers': 0.297662,
        'PlotArea(insqft)_imp': 0.480518,
        'CarpetArea_imp': 0.511952,
        'FAR': 0.031309,
        'Direction Facing': 5.129946,
        'Property Ownership': 0.589478,
        'FlooringRating': 0.266187,
        'Total_No_of_Floors': 1.172212,
        'Floor_Category': 0.821942,
        'Property_Age_Enc': 3.677158,
        'Furnishing_Enc': 1.445594,
        'Loading_Factor_Status': 1.544964,
        'Swimming Pool_Y': 0.297662,
        'is_Spacious_Spacious': 0.995504,
        'City_Mumbai': 0.0,
    }

    if city == 'Mumbai':
        default_data['City_Mumbai'] = 1.0

    data = [scalers[key] if key in scalers else default_data[key] for key in default_data]
    x = np.concatenate((np.array(data), np.zeros(187)))

    if loc_index >= 0:
        x[loc_index] = 1

    return round(np.expm1(__model_buy.predict([x])[0]) / 100000, 2)


def get_rent_estimate(city, location, sqft, bhk, bath):
    other_features_n = 8

    try:
        loc_index = __location_indices["rent"].index(location) + other_features_n
    except:
        loc_index = -1

    scaler_default = {'Lift': 2.198836, 'BHK_imp': bhk, 'Bedrooms_imp': 1.574427, 'Bathrooms_imp': bath,
                      'No_of_Towers': 3.640649, 'SuperArea(insqft)_imp': 1514.815905, 'PlotArea(insqft)_imp': sqft,
                      'CarpetArea_imp': 6.989543}

    scaler_df = pd.DataFrame(scaler_default, index=[0])

    vals = __scaler["rent"].transform(scaler_df)[0]
    scalers = {key: vals[i] for i, key in enumerate(scaler_default)}

    default1 = {
        'Lift': 0.149391,
        'BHK_imp': 0.221715,
        'No_of_Towers': 0.038634,
        'PlotArea(insqft)_imp': 0.038634,
        'CarpetArea_imp': 0.067324,
        'Property Ownership': 0.546132,
        'Total No. of Floors': 0.881019,
        'listing_type': 4.917987,
        'FlooringRating': 0.214974,
        'Floor_Category': 0.959925,
        'Property_Age_Enc': 3.245418,
        'Furnishing_Enc': 1.025784,
        'Loading_Factor_Status': 1.415657,
    }

    default2 = {
        'City_Mumbai': 0.415346,
        'Vaastu Compliant_Y': 0.148493,
        'Central AC_Y': 0.093818,
        'Tennis Court_Y': 0.147251,
        'Power Backup_Y': 0.294501,
        'S & G Lift_Y': 0.116185,
        'Servant Rooms_Y': 0.090090,
        'Parking_Open_Y': 0.094129
    }

    if city == 'Mumbai':
        default2['City_Mumbai'] = 1.0

    data1 = [scalers[key] if key in scalers else default1[key] for key in default1]
    data2 = list(default2.values())
    x = np.concatenate((np.concatenate((np.array(data1), np.zeros(169))), np.array(data2)))

    if loc_index >= 0:
        x[loc_index] = 1

    return round(np.expm1(__model_rent.predict([x])[0]) / 1000, 2)


def get_estimated_price(city, location, sqft, bhk, bath):
    return [get_buy_estimate(city, location, sqft, bhk, bath),
            get_rent_estimate(city, location, sqft, bhk, bath)]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __location_indices
    global __scaler
    global __model_buy
    global __model_rent

    # with open("./artifacts/columns.json", "r") as f:

    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    json_url = os.path.join(SITE_ROOT, "artifacts", "columns.json")
    f = open(json_url)
    data = json.load(f)
    __data_columns = data['data_columns']
    __locations = data["cities"]

    __location_indices = {}
    __location_indices["buy"] = data['location_indices_buy']
    __location_indices["rent"] = data['location_indices_rent']

    import pickle

    __scaler = {}
    json_url = os.path.join(SITE_ROOT, "artifacts", "scaler_buy.sav")
    with open(json_url, 'rb') as f:
        __scaler["buy"] = pickle.load(f)

    json_url = os.path.join(SITE_ROOT, "artifacts", "scaler_rent.sav")
    with open(json_url, 'rb') as f:
        __scaler["rent"] = pickle.load(f)

    if __model_buy is None:
        json_url = os.path.join(SITE_ROOT, "artifacts", "Ridge_Reg.pkl")
        with open(json_url, 'rb') as f:
            __model_buy = pickle.load(f)

    if __model_rent is None:
        json_url = os.path.join(SITE_ROOT, "artifacts", "Voting_Reg.pkl")
        with open(json_url, 'rb') as f:
            __model_rent = pickle.load(f)

    print("loading saved artifacts...done")


def get_location_names(city):
    if city in __locations:
        return __locations[city]
    else:
        return ""


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
