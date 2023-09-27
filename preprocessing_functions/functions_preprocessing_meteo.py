import pandas as pd
import datetime
from preprocessing_functions.utils_preprocessing import return_monday

def farenheit_to_celsius(degrees_farenheit: float) -> float:

    degrees_celsius = (degrees_farenheit - 32) * (5/9)
    return degrees_celsius

def first_meteo_preprocess(meteo: pd.DataFrame, filter_dates: bool = False, date_min: datetime = None, date_max: datetime = None) -> pd.DataFrame:
    """
    First preprocessing of the output raw data from the scrappy code. Remove
    useless columns, convert the temperatures to celsius, parse dates and
    filter some strange dates that might appear (Especialy for the training).
    """
    
    if "Unnamed: 0" in meteo.columns:
        meteo.drop("Unnamed: 0", axis=1, inplace=True)

    meteo["Temp_max"] = meteo["Temp_max"].apply(farenheit_to_celsius)
    meteo["Temp_min"] = meteo["Temp_min"].apply(farenheit_to_celsius)
    meteo["Temp_avg"] = meteo["Temp_avg"].apply(farenheit_to_celsius)
    meteo["Dew_max"] = meteo["Dew_max"].apply(farenheit_to_celsius)
    meteo["Dew_avg"] = meteo["Dew_avg"].apply(farenheit_to_celsius)
    meteo["Dew_min"] = meteo["Dew_min"].apply(farenheit_to_celsius)

    meteo["Date"] = pd.to_datetime(meteo["Date"])

    if filter_dates:
        meteo = meteo.loc[(date_min <= meteo["Date"]) & (meteo["Date"] <= date_max)]

    return meteo

def meteo_groupby(meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Performs agrupation by week for all the meteorological measures 
    and returns a dataframe with metrics aggregated by week such
    as maximum temperature, maximum humidity, average wind etc.
    """

    # Convert the row Date to datetime:
    meteo["Date"] = pd.to_datetime(meteo["Date"])

    grouped_df = meteo.groupby(pd.Grouper(key="Date", freq="W")).agg(
                                    {
                                        'Temp_max':['max'],
                                        'Temp_avg':['mean', 'std'],
                                        'Temp_min':['min'],
                                        'Dew_max':['max'],
                                        'Dew_avg':['mean', 'std'],
                                        'Dew_min':['min'],
                                        'Hum_max':['max'],
                                        'Hum_avg':['mean', 'std'],
                                        'Hum_min':['min'],
                                        'Wind_max':['max'],
                                        'Wind_avg':['mean', 'std'],
                                        'Wind_min':['min'],
                                        'Pres_max':['max'],
                                        'Pres_avg':['mean', 'std'],
                                        'Pres_min':['min'],
                                        'Precipitation':['mean', 'sum'],
                                    }, axis=1
                                )

    grouped_df.columns = grouped_df.columns.map('_'.join).str.strip('_')
    grouped_df.reset_index(inplace=True)
    grouped_df = grouped_df.rename(columns={"Date": "dated"})

    # Get the dates to monday to do the join after:
    grouped_df["dated"] = grouped_df["dated"].apply(lambda x: return_monday(x))

    return grouped_df