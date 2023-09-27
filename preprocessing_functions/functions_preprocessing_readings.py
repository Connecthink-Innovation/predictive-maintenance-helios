import pandas as pd
from preprocessing_functions.utils_preprocessing import return_monday

# Maybe we will need this function in the future:

# def split_readings(readings: pd.DataFrame) -> tuple[pd.DataFrame]:
#     """
#     Split the readings dataset in four: powerReactive, powerActive, powerReactivePeak, powerActivePeak
#     """
    
#     powerReactive = readings.loc[readings["reading"] == "powerReactive"]
#     powerActive = readings.loc[readings["reading"] == "powerActive"]
#     powerReactivePeak = readings.loc[readings["reading"] == "powerReactivePeak"]
#     powerActivePeak = readings.loc[readings["reading"] == "powerActivePeak"]

#     return (powerReactive, powerActive, powerReactivePeak, powerActivePeak)


def pre_power(power: pd.DataFrame, type_reading: str) -> pd.DataFrame:
    """
    Preprocessing of the datasets powerReactive and powerActive. Takes as 
    arguments the readings and the type: Active or Reactive
    """
    # to date:
    power["dated"] = pd.to_datetime(power["dated"])

    # Eliminate useless columns:
    power.drop(["componentid", "reading"], axis=1, inplace=True)

    # separate the dataframe by components and drop the column component beacuse it does not give info anymore:
    sum_df = power.loc[power["component"] == "sum"].drop("component", axis=1)
    p1_df = power.loc[power["component"] == "p1"].drop("component", axis=1)
    p2_df = power.loc[power["component"] == "p2"].drop("component", axis=1)
    p3_df = power.loc[power["component"] == "p3"].drop("component", axis=1)

    if type_reading == "Active":
        # Change the name of the column "value" to "value_" + "measure" so when we do the merge later
        sum_df.rename(columns = {"value": "powerActive_sum"}, inplace=True)
        p1_df.rename(columns = {"value": "powerActive_p1"}, inplace=True)
        p2_df.rename(columns = {"value": "powerActive_p2"}, inplace=True)
        p3_df.rename(columns = {"value": "powerActive_p3"}, inplace=True)
       
    if type_reading == "Reactive":
        # Change the name of the column "value" to "value_" + "measure" so when we do the merge later
        sum_df.rename(columns = {"value": "powerReactive_sum"}, inplace=True)
        p1_df.rename(columns = {"value": "powerReactive_p1"}, inplace=True)
        p2_df.rename(columns = {"value": "powerReactive_p2"}, inplace=True)
        p3_df.rename(columns = {"value": "powerReactive_p3"}, inplace=True)

    # Group for and drop the column
    sum_df = sum_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p1_df = p1_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p2_df = p2_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p3_df = p3_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()

    # Dates to Monday and to string so we have no errors when doing joins:
    sum_df["dated"] = sum_df["dated"].apply(lambda x: return_monday(x))
    sum_df["dated"] = sum_df["dated"].apply(lambda x: str(x))

    p1_df["dated"] = p1_df["dated"].apply(lambda x: return_monday(x))
    p1_df["dated"] = p1_df["dated"].apply(lambda x: str(x))

    p2_df["dated"] = p2_df["dated"].apply(lambda x: return_monday(x))
    p2_df["dated"] = p2_df["dated"].apply(lambda x: str(x))

    p3_df["dated"] = p3_df["dated"].apply(lambda x: return_monday(x))
    p3_df["dated"] = p3_df["dated"].apply(lambda x: str(x))

    # add all the mesures in one dataframe
    df = pd.merge(sum_df, p1_df, on=["dated", "id"])
    df = pd.merge(df, p2_df, on=["dated", "id"])
    df = pd.merge(df, p3_df, on=["dated", "id"])

    return df

def pre_power_peak(power_peak: pd.DataFrame, type_reading: str) -> pd.DataFrame:
    """
    Preprocessing of the datasets powerReactivePeak and powerActivePeak. Takes as 
    arguments the readings and the type: Active or Reactive
    """

    power_peak["dated"] = pd.to_datetime(power_peak["dated"])

    # Eliminate useless columns:
    power_peak.drop(["componentid", "reading"], axis=1, inplace=True)

    if type_reading == "Active":
        # Change the name of the column "value" to "ActivePeak"
        power_peak.rename(columns = {"value": "ActivePeak"}, inplace=True)
    
    if type_reading == "Reactive":
        # Change the name of the column "value" to "ReactivePeak"
        power_peak.rename(columns = {"value": "ReactivePeak"}, inplace=True)
    
    # Group:
    power_peak = power_peak.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()

    # Dates to Monday and to string so we have no errors when doing joins:
    power_peak["dated"] = power_peak["dated"].apply(lambda x: return_monday(x))
    power_peak["dated"] = power_peak["dated"].apply(lambda x: str(x))

    return power_peak