import pickle
import pandas as pd
import argparse
import ast
from preprocessing_functions.utils_preprocessing import return_monday
from predictive_models.predictive_models_functions import make_predictions_lights, make_predictions_eboxes
from preprocessing_functions.functions_preprocessing_alarms import first_eboxes_preprocess, first_lights_preprocess, big_preprocess_eboxes, big_preprocess_lights
from preprocessing_functions.functions_preprocessing_readings import pre_power, pre_power_peak
from preprocessing_functions.functions_preprocessing_meteo import first_meteo_preprocess, meteo_groupby
from preprocessing_functions.functions_joins import join_light_alarms_readings_meteo, join_eboxes_alarms_readings_meteo

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("--di", "--raw_data_folders_directory", help="Provide the local directory on your machine of the folders raw_data and meteo_raw_data. Example: /home/leibniz/Desktop/IHMAN")
# About this dates it will be better to detect the current day where we are (sunday or any other day from the week) and get the data bia sql from the database. For now passing the dates as an argument
# is how we will get the date ranges but it is a temporal fix
parser.add_argument("--da", "--min_max_dates", help="Provide the min and max dates of the data you are doing predictions", nargs="+", default=[])
parser.add_argument("--mo", "--model", help="Indicate the model you want to use for predictions of lights. 'adboc' for a model that does not use the readings or 'default' for a model that does use the readings", default="default")
parser.add_argument("--met", "--meteo", help="Indicate using a boolean True or False if you want to use meteorological data", default="True")

args = parser.parse_args()

# Raise an error in case that the user has not specified the date range:
if len(args.da) == 0:
    raise Exception("Error: You have to specify the date range of your data")

# Verify that the dates are indeed the Monday from the first date and the Sunday of the last week:
if pd.to_datetime(args.da[0]).weekday() != 0:
    raise Exception("Error: The first date that you have specified in the arguments is not a Monday")
if pd.to_datetime(args.da[1]).weekday() != 6:
    raise Exception("Error: The second date that you have specified in the arguments is not a Sunday")

data_dir = args.di
predicting_min = args.da[0]
predicting_max = args.da[1]
model_type = args.mo
use_meteo_data = ast.literal_eval(args.met)

print("Inputed date range:")
print(predicting_min)
print(predicting_max)

print("Meteo selected: " + str(use_meteo_data))

# read data:
eboxes_alarms = pd.read_csv(f"{data_dir}/data_to_predict/eboxes_alarms.csv")
light_alarms = pd.read_csv(f"{data_dir}/data_to_predict/lights_alarms.csv")
nodes = pd.read_csv(f"{data_dir}/data_to_predict/nodes.csv")
powerActive = pd.read_csv(f"{data_dir}/data_to_predict/powerActive.csv")
powerReactive = pd.read_csv(f"{data_dir}/data_to_predict/powerReactive.csv")
powerActivePeak = pd.read_csv(f"{data_dir}/data_to_predict/powerActivePeak.csv")
powerReactivePeak = pd.read_csv(f"{data_dir}/data_to_predict/powerActivePeak.csv")

# For meteo data we have two options. In case that ths user has selected meteo for the model import the scrapped data.
# In case that the user has selected NOT to use meteo data we will dowload a default dataframe that has proven not to affect much the probability of breakdown:
if use_meteo_data:
    meteo = pd.read_csv("scrapped_meteo_data/meteo_data.csv")

    # We only preprocess the meteo data in case that the user has sleected to use the scraped meteo data
    meteo = first_meteo_preprocess(
        meteo = meteo,
        filter_dates = False
    )
    meteo = meteo_groupby(meteo)
else:
    # In case that the user has selected NOT using the meteo data, the code will use the default meteo data that does not affect probabilities
    # There is no need for preprocessing in this case as it will be done directly in the functions join_eboxes_alarms_readings_meteo and
    #  join_eboxes_alarms_readings_meteo
    meteo = pd.read_csv("default_meteo/default_meteo.csv")

# Almost the same preprocessing as we do on preparing the data for training with some minor changes.
# For example in the alarms preprocessing functions we have to specify the argument for_predicting

powerReactive, powerActive = pre_power(powerReactive, "Reactive"), pre_power(powerActive, "Active")
powerReactivePeak, powerActivePeak = pre_power_peak(powerReactivePeak, "Reactive"), pre_power_peak(powerActivePeak, "Active")

eboxes_alarms = first_eboxes_preprocess(eboxes_alarms, for_predicting=True)

# Here we want to create a boolean variable that indicates if the dataframe eboxes_alarms is empty.
# This can happen when there are no registered "brdpower" errors in the last 5 weeks. If we don't 
# take into account this, it can happend that you end up feeding the function big_preprocess an empty
# dataframe wich will result in an error.
eboxes_empty = (len(eboxes_alarms) == 0)

if not eboxes_empty:
    eboxes_alarms = big_preprocess_eboxes(
        eboxes_alarms = eboxes_alarms,
        for_predicting = True,
        predicting_min_date = predicting_min,
        predicting_max_date = predicting_max
    )

    eboxes_alarms = join_eboxes_alarms_readings_meteo(
        eboxes_alarms = eboxes_alarms,
        eboxes_powerReactivePeak = powerReactivePeak,
        eboxes_powerReactive = powerReactive,
        eboxes_powerActive = powerActive,
        eboxes_powerActivePeak = powerActivePeak,
        use_meteo_data = use_meteo_data,
        meteo = meteo
    )
else:
    # If the eboxes_alarms dataframe is indeed empty we will have errors in a few lines when creating
    # the dataframe for the non_error_eboxes. This error will appear because for creating the non_error_eboxes
    # dataframe we use the weeks from the dataframe eboxes_alarms (eboxes_alarms["week-i"][0]). In the case 
    # that the dataframe is empty we have not executed the function big_preprocess_eboxes and the columns
    # "week-i" do not exist

    # The solution will simply be to generate the dates with a date range and create new columns for the
    # empty eboxes_alarms dataframe with the dates generated so the code can take this dates for generating
    # the non_error_eboxes dataframe.

    dates_range = pd.date_range(start=predicting_min, end=predicting_max, freq="W")
    dates_range = [str(return_monday(elem)) for elem in dates_range]

    # Rewrite the dataframe to use it later. Note that we add too an id column for in a few lines generating
    # the non_error_eboxes_nodes .
    eboxes_alarms = pd.DataFrame(
        {"id": [None], "week-4": [dates_range[0]], "week-3": [dates_range[1]], "week-2": [dates_range[2]], "week-1": [dates_range[3]], "current_week": [dates_range[4]]}
    )

    
# Untill this point we have just considered the eboxes that have had an error in the last 4 weeks. 
# We should consider too all the other alarms that have no errors in the last 4 weeks and put them
# in a dataframe format identical to the eboxes-alarms dataframe. This way we will be able to do predictionns
# for all the eboxes in the city.

# Let's get the nodes that have not suffered any alarms in the last weeks:
eboxes_nodes = nodes.loc[nodes["type"] == "box"]
non_error_eboxes_nodes = eboxes_nodes.loc[~eboxes_nodes["id"].isin(eboxes_alarms["id"])]

# Now we have to build a dataframe with the same structure as light_eboxes for this non_error_eboxes:
non_error_eboxes = pd.DataFrame(
    {
        "id": non_error_eboxes_nodes["id"],
        "week-4": eboxes_alarms["week-4"][0],
        "hours_week-4": 0,
        "week-3": eboxes_alarms["week-3"][0],
        "hours_week-3": 0,
        "week-2": eboxes_alarms["week-2"][0],
        "hours_week-2": 0,
        "week-1": eboxes_alarms["week-1"][0],
        "hours_week-1": 0,
        "current_week": eboxes_alarms["current_week"][0],
        "hours_current_week": 0,
        "lat": non_error_eboxes_nodes["lat"],
        "lon": non_error_eboxes_nodes["lon"]
    }
)
# Create the dataframe ready for the model:

non_error_eboxes = join_eboxes_alarms_readings_meteo(
    eboxes_alarms = non_error_eboxes,
    eboxes_powerReactivePeak = powerReactivePeak,
    eboxes_powerReactive = powerReactive,
    eboxes_powerActive = powerActive,
    eboxes_powerActivePeak = powerActivePeak,
    use_meteo_data = use_meteo_data,
    meteo = meteo
)

# Now for the the lights:

light_alarms = first_lights_preprocess(light_alarms, for_predicting=True)

# In the same way as the eboxes, we create a boolean variable to control if the light_alarms dataframe is empty
lights_empty = (len(light_alarms) == 0)

if not lights_empty:
    light_alarms = big_preprocess_lights(
        light_alarms = light_alarms,
        for_predicting = True,
        predicting_min_date = predicting_min,
        predicting_max_date = predicting_max
    )

    light_alarms = pd.merge(light_alarms, nodes, on="id", how="left")

    # TO BE DELETED:
    light_alarms.to_csv("tests_folder_TOBEDELETED/light_alarms.csv")
    meteo.to_csv("tests_folder_TOBEDELETED/meteo.csv")

    light_alarms = join_light_alarms_readings_meteo(
        light_errors = light_alarms,
        eboxes_powerReactivePeak = powerReactivePeak,
        eboxes_powerReactive = powerReactive,
        eboxes_powerActive = powerActive,
        eboxes_powerActivePeak = powerActivePeak,
        use_meteo_data = use_meteo_data,
        meteo = meteo
    )

    # TO BE DELETED:
    light_alarms.to_csv("tests_folder_TOBEDELETED/final.csv")

else:
    # The same that we did for the eboxes must be done for the lights:

    dates_range = pd.date_range(start=predicting_min, end=predicting_max, freq="W")
    dates_range = [str(return_monday(elem)) for elem in dates_range]

    # Rewrite the dataframe to use it later. Note that we add too an id column for in a few lines generating
    # the non_error_eboxes_nodes .
    light_alarms = pd.DataFrame(
        {"id": [None], "week-4": [dates_range[0]], "week-3": [dates_range[1]], "week-2": [dates_range[2]], "week-1": [dates_range[3]], "current_week": [dates_range[4]]}
    )

# Untill this point we have just considered the lights that have had an error in the last 4 weeks. 
# We should consider too all the other alarms that have no errors in the last 4 weeks and put them
# in a dataframe format identical to the light-alarms dataframe. This way we will be able to do predictionns
# for all the lights in the city.

# Let's get the nodes that have not suffered any alarms in the last weeks:
light_nodes = nodes.loc[nodes["type"] == "light"]
non_error_lights_nodes = light_nodes.loc[~light_nodes["id"].isin(light_alarms["id"])]

# Now we have to build a dataframe with the same structure as light_alarms for this non_error_lights:
non_error_lights = pd.DataFrame(
    {
        "id": non_error_lights_nodes["id"],
        "week-4": light_alarms["week-4"][0],
        "hours_week-4": 0,
        "week-3": light_alarms["week-3"][0],
        "hours_week-3": 0,
        "week-2": light_alarms["week-2"][0],
        "hours_week-2": 0,
        "week-1": light_alarms["week-1"][0],
        "hours_week-1": 0,
        "current_week": light_alarms["current_week"][0],
        "hours_current_week": 0,
        "ebox_id": non_error_lights_nodes["ebox_id"],
        "lat": non_error_lights_nodes["lat"],
        "lon": non_error_lights_nodes["lon"]
    }
)
# Create the dataframe ready for the model:

non_error_lights = join_light_alarms_readings_meteo(
    light_errors = non_error_lights,
    eboxes_powerReactivePeak = powerReactivePeak,
    eboxes_powerReactive = powerReactive,
    eboxes_powerActive = powerActive,
    eboxes_powerActivePeak = powerActivePeak,
    use_meteo_data = use_meteo_data,
    meteo = meteo
)

# Generate the predictions:
if not lights_empty:
    print("Predictions for luminarire with errors in the last weeks:")
    make_predictions_lights(light_alarms, model_type)
else:
    print("There have been no lighterr or lightcomm errors registered in the last 5 weeks")

print("Predictions for luminarire with no errors in the last weeks:")
make_predictions_lights(non_error_lights, model_type)

if not eboxes_empty:
    print("Predictions for eboxes with errors in the last weeks:")
    make_predictions_eboxes(eboxes_alarms)
else:
    print("There have been no brdpower errors registered in the last 5 weeks")

print("Predictions for eboxes with no errors in the last weeks:")
make_predictions_eboxes(non_error_eboxes)

