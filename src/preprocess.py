import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(data):
    # Remove unnecessary column
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # Convert categorical columns
    data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
    data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
    data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})

    return data

def get_features_target(data):
    X = data[[
        "Age",
        "GraduateOrNot",
        "AnnualIncome",
        "FamilyMembers",
        "ChronicDiseases",
        "FrequentFlyer",
        "EverTravelledAbroad"
    ]]
    
    y = data["TravelInsurance"]

    return X, y