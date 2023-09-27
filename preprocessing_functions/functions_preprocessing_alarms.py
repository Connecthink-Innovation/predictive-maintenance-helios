import pandas as pd
import time
import datetime
from preprocessing_functions.utils_preprocessing import return_monday

def first_lights_preprocess(light_alarms: pd.DataFrame, for_predicting = False, min_date = "1982-01-04", max_date = "2023-04-04") -> pd.DataFrame:
    """
    Performs basic preprocessing on the raw data of alarms such as filtering the errors that
    we want to predict ("lightcomm", "lighterr"), remove useless columns, parse dates and 
    eliminate dates out of a defined range.
    """

    light_alarms = light_alarms.loc[(light_alarms["alarm"] == "lightcomm") | ((light_alarms["alarm"] == "lighterr"))]

    light_alarms = light_alarms[["id", "dated", "alarm", "flag"]]

    light_alarms["dated"] = pd.to_datetime(light_alarms["dated"])

    if not for_predicting:
        # We have to do some filtering for the raw data because in the datasets for training we have some nonsense data
        # like data from the future.
        light_alarms = light_alarms.loc[(min_date <= light_alarms["dated"]) & (light_alarms["dated"] <= max_date)]

    return light_alarms

def first_eboxes_preprocess(eboxes_alarms: pd.DataFrame, for_predicting = False, min_date = "1982-01-04", max_date = "2023-04-04") -> pd.DataFrame:
    """
    Performs basic preprocessing on the raw data of alarms such as filtering the errors that
    we want to predict ("lightcomm", "lighterr"), remove useless columns, parse dates and 
    eliminate dates out of a defined range.
    """

    eboxes_alarms = eboxes_alarms.loc[eboxes_alarms["subtype"] == "brdpower"]

    eboxes_alarms = eboxes_alarms[["id", "dated", "subtype", "flag"]]

    eboxes_alarms["dated"] = pd.to_datetime(eboxes_alarms["dated"])

    if not for_predicting:
        # We have to do some filtering for the raw data because in the datasets for training we have some nonsense data
        # like data from the future.
        eboxes_alarms = eboxes_alarms.loc[(min_date <= eboxes_alarms["dated"]) & (eboxes_alarms["dated"] <= max_date)]

    return eboxes_alarms

def big_preprocess_lights(light_alarms: pd.DataFrame, predicting_min_date: str = None, predicting_max_date: str = None, for_predicting: bool = False) -> pd.DataFrame:
    """
    The function converts the data of the alarms into a dataframe that is usable for training the models.
    Review the coments on the code to understand step by step what the code does.
    """

    start_time = time.time()

    # We create a dataframe of each one of the ids and we will clean the alarms. The final objective is to calculate
    # the percentage of time the lampost is not working wich is the time that passes from an alarm "on" utill it is 
    # turned "off". In the dataframe we have cases in wich there are two consecutive "on" or "off" alarms so we have
    # filter and just keep the first "on" and the first "off" in this case so we get the real time the lampost has been
    # not functioning. In the case that the alarm is turned "on" during a week and it is not turned "off" untill the next
    # week we will insert a fake "off" alarm at the last moment of the week and we will turn it "on" again in the first moment
    # of the following week.

    # The first step will be to dowload the data:
    # Read recipe inputs

    df = light_alarms.copy()

    # Sort by id and date of the alarm:
    df["dated"] = pd.to_datetime(df["dated"])
    df = df.sort_values(["id", "dated"])
    # As far as we know if an alarm is "set" it means it is "off" so to keep things simple we will sustitute the
    # "set" values with "off"
    df.replace(to_replace="set", value="off", inplace=True)

    # Generate all the weeks from the first day we have data untill the last day. We will need this weeks later to
    # do a left join with the weeks that there are alarms

    # If we are preprocessing the data for doing predictions we will want to add the date_min and date max manualy,
    # otherwise the function will only consider the dates of the errors and you may end up with nan rows
    if for_predicting: 
        start_date = pd.to_datetime(predicting_min_date)
        end_date = pd.to_datetime(predicting_max_date)
    else:
        start_date = df["dated"].min()
        end_date = return_monday(df["dated"].max()) + pd.Timedelta(days=7)
    
    weeks = pd.date_range(start=start_date, end=end_date, freq='W').floor("D").strftime('%Y-%m-%d %H:%M:%S').tolist()
    # Transform to a dataframe to do operations later:
    weeks = pd.DataFrame(weeks, columns=["week"])
    weeks["week"] = pd.to_datetime(weeks["week"])
    weeks["week"] = weeks["week"].apply(lambda x: return_monday(x))
    # For each unique id we will implement all the code:
    # First create the dataframe where we will store all the resturned data:
    general_lag_dataframe = pd.DataFrame()

    # List of all the ids:
    ids_list = df["id"].unique()

    # Store the name of the columns in oder to later reordenate the final returned dataframe.
    if for_predicting:
        columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
                        "current_week", "hours_current_week"]
    else:     
        columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
                        "current_week", "hours_current_week", "week+1", "hours_week+1", "week+2", "hours_week+2", "week+3", "hours_week+3", "week+4", "hours_week+4"]

    # Generate a dataframe for each one of the ids and then apply all the transformations. Once done, add the data
    # to the dataframe general_lag_dataframe.
    for idd in ids_list:
        print(idd)
        
        # Dataframe of all the alarms for the id "idd":
        tt = df.loc[df["id"] == idd]
        
        #For each one of the elements in the list we have: elem[0] contains the week represented by sunday 
        # and elem[1] has the data in a dataframe
        grouped_weeks = list(tt.groupby(pd.Grouper(key="dated", freq="W")))
        # Here we will transform all the tuples to lists so we can change the elements after
        grouped_weeks = [list(elem) for elem in grouped_weeks]

        # Now we have to filtrate the empty dataframes that generates groupby for the weeks where there are no alarms
        grouped_weeks = [elem for elem in grouped_weeks if not elem[1].empty]

        # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on" for the first element
        # of the list grouped_weeks. We do it because in the iteration we will not consider it so we have to do it now
        first_week_data = grouped_weeks[0][1]
        first_week_data["prev_flag"] = first_week_data["flag"].shift()

        first_week_data = first_week_data.loc[
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"] == "off")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"] == "on")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"].isna()))
        ]

        # We store it to the data of the first week:
        grouped_weeks[0][1] = first_week_data

        # Here we begin the iteration for all the other weeks:
        for i, (week, data) in enumerate(grouped_weeks[1:], 1): # We do not consider the first week because it has no previous week to check
            # Get the data and week from the previous week:
            previous_week, previous_week_data = grouped_weeks[i-1][0], grouped_weeks[i-1][1]

            # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on"
            data["prev_flag"] = data["flag"].shift()
            data = data.loc[
                ((data["flag"] == "on") & (data["prev_flag"] == "off")) |
                ((data["flag"] == "off") & (data["prev_flag"] == "on")) |
                ((data["flag"] == "off") & (data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
                ((data["flag"] == "on") & (data["prev_flag"].isna()))
            ]

            # Data that we will need:
            last_moment_previous_week = previous_week + pd.Timedelta(hours=23, minutes=59, seconds=59) #We will need the last moment of the week
            first_moment_current_week = (week - pd.Timedelta(days=week.dayofweek)).replace(hour=0, minute=0, second=0) #We will need to the first moment of the current week

            #We have to get the last flag of the previous week
            last_flag_previous_week = previous_week_data.loc[previous_week_data.index[-1], "flag"]

            if last_flag_previous_week == "on":
                # If the last flag from the previous week is "on" then we have to set a new row on the previous week data
                # in the last position to set a flag "off". Then, in the current week data we will add a new row before the
                # first week to set again the alarm to "on"

                # Here we create the new row to set the alarm "off" in the previous week
                new_row_previous_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [last_moment_previous_week],
                    "alarm": ["turn_off_end_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["off"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very end of the week
                new_previous_week_data = pd.concat(
                    [previous_week_data, new_row_previous_week],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i-1][1] = new_previous_week_data

                # Now we have to set the flag "on" in the first moment of the current week:
                # Here we create the new row to set the alarm "on" in the current week
                new_row_current_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [first_moment_current_week],
                    "alarm": ["turn_on_begining_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["on"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very begining of the week
                new_current_week_data = pd.concat(
                    [new_row_current_week, data],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i][1] = new_current_week_data
            else:
                # Simply update the dataframe with the same but with removed rows that contain two identical flags in a row
                grouped_weeks[i][1] = data

        # Once this is done we have to check if the last alarm of the last week is "on". In this case we will add a row turning it off
        # in the last moment of the week:
        last_recorded_week, last_recorded_week_data = grouped_weeks[-1][0], grouped_weeks[-1][1]

        # Get the last flag from the last week
        last_flag = last_recorded_week_data["flag"].values[-1]
        # and get the last moment of the last week
        last_moment_last_recorded_week = last_recorded_week + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if last_flag == "on":
            new_row_last_flag = pd.DataFrame(
                {
                "id": [idd],
                "dated": [last_moment_last_recorded_week],
                "alarm": ["turn_off_end_last_week"], # We will put this in alarm to know witch alarms where inserted by us
                "flag": ["off"],
                "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                }
            )

            last_recorded_week_data = pd.concat(
                [last_recorded_week_data, new_row_last_flag],
                sort=True
            )

            grouped_weeks[-1][1] = last_recorded_week_data

        # Now we have to concat all the dataframes from all the weeks contained in the list grouped_weeks in one big dataframe
        concatenated_weeks = pd.concat(
            [week_data[1] for week_data in grouped_weeks],
            sort=True # Remove the warning of pd.concat
        )

        # At this point there are some cases where we will still have two "on" alarms or two "off" alarms in a row. For example
        # in the case that we have two weeks in a row where we only have "on" alarms utill now the code is going to return the 
        # begining of the end of the previous week with "off", the beggining of the current week with "off" and before the "off"
        # of the end of the week we will still have an "on" alarm. This is caused because the deletion of the same alarms in a row is done 
        # before the add of the new rows in the beggining and end of the week

        # So let's eliminate this cases too:
        # first we have to frop the old prev_flag column that now is useless:
        concatenated_weeks.drop("prev_flag", axis=1, inplace=True)

        # and create the new one:
        concatenated_weeks["prev_flag_concat"] = concatenated_weeks["flag"].shift()

        concatenated_weeks = concatenated_weeks.loc[
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"] == "off")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"] == "on")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"].isna()))
        ]

        # We have some weeks where there is no data and thus if this weeks are between an "on" and "off" flag they will not
        # appear on the dataframe. What we will do is a left join with the variable week generated at the begining
        # of the notebook to see the rows that do not appear. Then we will activate the alarm at the begining of the week
        # and deactivate it at the end of the same week:

        # Set the day to monday
        concatenated_weeks["week"] = concatenated_weeks["dated"].apply(lambda x: return_monday(x))
        # Left join with weeks to detect the missing weeks
        concatenated_weeks_merged = pd.merge(weeks, concatenated_weeks, on="week", how="left")

        # Store the result of the filling:
        filled_dataframe = pd.DataFrame()

        # We add the first row
        if not pd.isna(concatenated_weeks_merged.iloc[0]["alarm"]):
            first_new_row = pd.DataFrame(
                {
                    "week": [concatenated_weeks_merged.iloc[0]["week"]],
                    "alarm": [concatenated_weeks_merged.iloc[0]["alarm"]],
                    "dated": [concatenated_weeks_merged.iloc[0]["dated"]],
                    "flag": [concatenated_weeks_merged.iloc[0]["flag"]],
                    "id": [concatenated_weeks_merged.iloc[0]["id"]]
                }
            )

            filled_dataframe = pd.concat(
                [filled_dataframe, first_new_row],
                sort=True
            )

        # Begin the iteration where we will add one by one the rows to the dataframe filled_dataframe
        for i in range(1, len(concatenated_weeks_merged)-1):

            current_row = concatenated_weeks_merged.iloc[i]

            # If we find a normal row we add it to the dataframe
            if (current_row["alarm"] == "lightcomm") | (current_row["alarm"] == "lighterr") | (current_row["alarm"] == "turn_off_end_week") | (current_row["alarm"] == "turn_on_begining_week"):
                current_row["dated"] = pd.to_datetime(current_row["dated"])

                new_row = pd.DataFrame(
                    {
                        "week": [current_row["week"]],
                        "alarm": [current_row["alarm"]],
                        "dated": [current_row["dated"]],
                        "flag": [current_row["flag"]],
                        "id": [current_row["id"]]
                    }
                )
                filled_dataframe = pd.concat([filled_dataframe, new_row], sort=True)

            if not filled_dataframe.empty:
                # If the last row of the filled dataframe is a "turn_off_week" ant the current is a Emty we add the "on" and
                # off for the begining and the end of the week
                last_row_filled_dataframe = filled_dataframe.iloc[-1]

                if (pd.isna(current_row["alarm"])) & (last_row_filled_dataframe["alarm"] in ["turn_off_end_week", "turn_off_end_week_filled"]):

                    new_row_begining_dated = datetime.datetime.combine(current_row["week"], datetime.time(0,0,0))
                    new_row_begining_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_on_begining_week_filled"],
                            "dated": [new_row_begining_dated], # Add the time to the date
                            "flag": ["on"],
                            "id": [idd]
                        }
                    )
                    # We have to add 6 days because the representative of the week in this case is 
                    new_row_end_dated = datetime.datetime.combine(current_row["week"]+ pd.Timedelta(days=6), datetime.time(23,59,59))
                    new_row_end_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_off_end_week_filled"],
                            "dated": [new_row_end_dated], # In this case we add the time to represent the last moment of the week
                            "flag": ["off"],
                            "id": [idd]
                        }
                    )

                    filled_dataframe = pd.concat(
                        [filled_dataframe, new_row_begining_week, new_row_end_week],
                        sort=True
                    )


        # Again we have to do a left join with the dataframe weeks to detect the NaN values:
        filled_dataframe_merged = pd.merge(weeks, filled_dataframe, on="week", how="left")

        # At this point there are some cases where the last alarm of the dataframe is an "on" so we have to turn the alarm off at the end of the week:
        if filled_dataframe_merged["flag"].iloc[-1] == "on": # Check if the last alarm is still on
            
            # get the last moment of the week
            end_of_week = pd.to_datetime(filled_dataframe_merged["week"].iloc[-1]) + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)

            row_to_add = pd.DataFrame(
                {
                    "week": [filled_dataframe_merged["week"].iloc[-1]],
                    "alarm": ["turn_off_end_week"],
                    "dated": end_of_week,
                    "flag": ["off"],
                    "id": idd
                }
            )

            filled_dataframe_merged = pd.concat(
                [filled_dataframe_merged, row_to_add],
                sort=True
            )
        
        # Now for each one of the weeks we have to calculate the total time passed between an "on" alarm and an "off" alarm

        # In this dataframe we will store the amount of hours of a light that has been malfunctioning for each week
        week_hours_dataframe = pd.DataFrame()
        for week in weeks["week"]:
            # In this variable we will store the amount of hours for this week:
            total_hours = 0
            on_timestamp = None

            # Dataframe with the alarms of the week:
            week_alarms_dataframe = filled_dataframe_merged.loc[filled_dataframe_merged["week"] == week]
            # Iterate trough the df to count the hours:
            for _, row in week_alarms_dataframe.iterrows():
                if row["flag"] == "on":
                    on_timestamp = row["dated"]

                if (row["flag"] == "off") & (on_timestamp is not None):
                    total_hours += (row["dated"] - on_timestamp).total_seconds() / 3600

                    on_timestamp = None

            new_week_hours = pd.DataFrame(
                {
                    "id": [idd],
                    "week": [week],
                    "malfunctioning_hours": [total_hours] 
                }
            )

            week_hours_dataframe = pd.concat(
                [week_hours_dataframe, new_week_hours],
                sort=True
            )

        # Now we want to get the data in the format:
        # row = {"week-4": date_week_prev_4, "hours_week-4": hours_week_prev_4, ..., "week-1": date_week_prev_1, "hours_week-1": hours_week_prev_1, "current_week": date_current_week, "hours_current_week":  "week+1": date_week_next_1, "hours_week+1": hours_week_next_1 ..., "week+4": date_week_next_4, "hours_week+4": hours_week_next_4}

        # In this dataframe we will store the data in the format we have mentioned:
        lag_dataframe = pd.DataFrame()
        
        # Here we have to follow two different paths. The first one will be when preparing the data for training and the other one will be when 
        # preparing the data for predicting. We shall begin with predicting:

        if for_predicting:
            lag_dataframe = pd.DataFrame(
                {
                    "id": [idd],

                    "week-4": [week_hours_dataframe.iloc[0]["week"]],
                    "hours_week-4": [week_hours_dataframe.iloc[0]["malfunctioning_hours"]],

                    "week-3": [week_hours_dataframe.iloc[1]["week"]],
                    "hours_week-3": [week_hours_dataframe.iloc[1]["malfunctioning_hours"]],

                    "week-2": [week_hours_dataframe.iloc[2]["week"]],
                    "hours_week-2": [week_hours_dataframe.iloc[2]["malfunctioning_hours"]],

                    "week-1": [week_hours_dataframe.iloc[3]["week"]],
                    "hours_week-1": [week_hours_dataframe.iloc[3]["malfunctioning_hours"]],

                    "current_week": [week_hours_dataframe.iloc[4]["week"]],
                    "hours_current_week": [week_hours_dataframe.iloc[4]["malfunctioning_hours"]]
                }
            )

            # Reorder the dataframe so it is in the same order we have defined:
            lag_dataframe = lag_dataframe[columns_order]
            
            # Add the dataframe to the general one:
            general_lag_dataframe = pd.concat(
                [general_lag_dataframe, lag_dataframe],
                sort=True
            )
        
        else:
            # Begin the loop at 4 and end at -4 so we don't get the error: "Out of range"
            for i in range(4, len(weeks)-4):
                
                # Create the new row to add:
                to_add_row = pd.DataFrame(
                    {
                        "id": [idd],

                        "week-4": [week_hours_dataframe.iloc[i-4]["week"]],
                        "hours_week-4": [week_hours_dataframe.iloc[i-4]["malfunctioning_hours"]],

                        "week-3": [week_hours_dataframe.iloc[i-3]["week"]],
                        "hours_week-3": [week_hours_dataframe.iloc[i-3]["malfunctioning_hours"]],

                        "week-2": [week_hours_dataframe.iloc[i-2]["week"]],
                        "hours_week-2": [week_hours_dataframe.iloc[i-2]["malfunctioning_hours"]],

                        "week-1": [week_hours_dataframe.iloc[i-1]["week"]],
                        "hours_week-1": [week_hours_dataframe.iloc[i-1]["malfunctioning_hours"]],

                        "current_week": [week_hours_dataframe.iloc[i]["week"]],
                        "hours_current_week": [week_hours_dataframe.iloc[i]["malfunctioning_hours"]],

                        "week+1": [week_hours_dataframe.iloc[i+1]["week"]],
                        "hours_week+1": [week_hours_dataframe.iloc[i+1]["malfunctioning_hours"]],

                        "week+2": [week_hours_dataframe.iloc[i+2]["week"]],
                        "hours_week+2": [week_hours_dataframe.iloc[i+2]["malfunctioning_hours"]],

                        "week+3": [week_hours_dataframe.iloc[i+3]["week"]],
                        "hours_week+3": [week_hours_dataframe.iloc[i+3]["malfunctioning_hours"]],

                        "week+4": [week_hours_dataframe.iloc[i+4]["week"]],
                        "hours_week+4": [week_hours_dataframe.iloc[i+4]["malfunctioning_hours"]]
                    }
                )

                lag_dataframe = pd.concat(
                    [lag_dataframe, to_add_row],
                    sort=True,
                    ignore_index=True
                )

            # Reorder the dataframe so it is in the same order we have defined:
            lag_dataframe = lag_dataframe[columns_order]
            
            # Add the dataframe to the general one:
            general_lag_dataframe = pd.concat(
                [general_lag_dataframe, lag_dataframe],
                sort=True
            )
        
    # Reordenate with the list columns_order:
    general_lag_dataframe = general_lag_dataframe[columns_order]

    end_time = time.time()

    print("Execution time:" + str(end_time - start_time))

    return general_lag_dataframe

def big_preprocess_eboxes(eboxes_alarms: pd.DataFrame, predicting_min_date: str = None, predicting_max_date: str = None, for_predicting: bool = False) -> pd.DataFrame:
    """
    The function converts the data of the alarms into a dataframe that is usable for training the models.
    Review the coments on the code to understand step by step what the code does.
    """
    
    start_time = time.time()

    # We create a dataframe of each one of the ids and we will clean the alarms. The final objective is to calculate
    # the percentage of time the ebox is not working wich is the time that passes from an alarm "off" utill it is 
    # turned "on". In the dataframe we have cases in wich there are two consecutive "on" or "off" alarms so we have
    # filter and just keep the first "on" and the first "off" in this case so we get the real time the ebox has been
    # not functioning. In the case that the alarm is turned "on" during a week and it is not turned "off" untill the next
    # week we will insert a fake "off" alarm at the last moment of the week and we will turn it "on" again in the first moment
    # of the following week.

    # The first step will be to dowload the data:
    # Read recipe inputs
    df = eboxes_alarms.copy()

    # We will reuse the code of the light alarms for the eboxes: The main thing that we have to modify for the eboxes
    # is that for the alarm subtype brdpower the flag "off" means that the ebox is suffering a breakdown and untill the
    # alarm turns "on" we will consider that the ebox has been down. This is exactly opposite to what happened with the
    # lamposts.

    # The first step is to change the name of the column subtype. Even tough this columns means a subtype of alarm, in this
    # code and from now on we will rename it as alarm so we are able to replicate a similar code as the one used for the lights
    df.rename(columns={"subtype": "alarm"}, inplace=True)

    # There are four different flags in this case: {off, on, offM, onM} depending if the error has been detected digitaly or manualy.
    # we do not care so we fill change this categories to {on off}:
    df["flag"] = df["flag"].replace(["onM"], "on")
    df["flag"] = df["flag"].replace(["offM"], "off")

    # As we already said the flags in the eboxes are opposite as the one in the lamposts, so to be able to use the same
    # code we will just change the values "off" to "on" and the "on" to "off" and then apply the same exact code that
    # we used for the lamposts.
    df["flag"] = df["flag"].replace({'on': 'offf', 'off': 'on'}).replace({"offf": "off"})

    # Sort by id and date of the alarm:
    df["dated"] = pd.to_datetime(df["dated"])
    df = df.sort_values(["id", "dated"])

    # Generate all the weeks from the first day we have data untill the last day. We will need this weeks later to
    # do a left join with the weeks that there are alarms

    # If we are preprocessing the data for doing predictions we will want to add the date_min and date max manualy,
    # otherwise the function will only consider the dates of the errors and you may end up with nan rows
    if for_predicting: 
        start_date = pd.to_datetime(predicting_min_date)
        end_date = pd.to_datetime(predicting_max_date)
    else:
        start_date = df["dated"].min()
        end_date = return_monday(df["dated"].max()) + pd.Timedelta(days=7)

    weeks = pd.date_range(start=start_date, end=end_date, freq='W').floor("D").strftime('%Y-%m-%d %H:%M:%S').tolist()
    # Transform to a dataframe to do operations later:
    weeks = pd.DataFrame(weeks, columns=["week"])
    weeks["week"] = pd.to_datetime(weeks["week"])
    weeks["week"] = weeks["week"].apply(lambda x: return_monday(x))

    # For each unique id we will implement all the code:
    # First create the dataframe where we will store all the resturned data:
    general_lag_dataframe = pd.DataFrame()

    # List of all the ids:
    ids_list = df["id"].unique()

    # Store the name of the columns in oder to later reordenate the final returned dataframe.
    if for_predicting:
        columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
                        "current_week", "hours_current_week"]
    else:     
        columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
                        "current_week", "hours_current_week", "week+1", "hours_week+1", "week+2", "hours_week+2", "week+3", "hours_week+3", "week+4", "hours_week+4"]

    # Generate a dataframe for each one of the ids and then apply all the transformations. Once done, add the data
    # to the dataframe general_lag_dataframe.
    for idd in ids_list:
        print(idd)
        
        # Dataframe of all the alarms for the id "idd":
        tt = df.loc[df["id"] == idd]
        
        #For each one of the elements in the list we have: elem[0] contains the week represented by sunday 
        # and elem[1] has the data in a dataframe
        grouped_weeks = list(tt.groupby(pd.Grouper(key="dated", freq="W")))
        # Here we will transform all the tuples to lists so we can change the elements after
        grouped_weeks = [list(elem) for elem in grouped_weeks]

        # Now we have to filtrate the empty dataframes that generates groupby for the weeks where there are no alarms
        grouped_weeks = [elem for elem in grouped_weeks if not elem[1].empty]

        # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on" for the first element
        # of the list grouped_weeks. We do it because in the iteration we will not consider it so we have to do it now
        first_week_data = grouped_weeks[0][1]
        first_week_data["prev_flag"] = first_week_data["flag"].shift()

        first_week_data = first_week_data.loc[
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"] == "off")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"] == "on")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"].isna()))
        ]

        # We store it to the data of the first week:
        grouped_weeks[0][1] = first_week_data

        # Here we begin the iteration for all the other weeks:
        for i, (week, data) in enumerate(grouped_weeks[1:], 1): # We do not consider the first week because it has no previous week to check
            # Get the data and week from the previous week:
            previous_week, previous_week_data = grouped_weeks[i-1][0], grouped_weeks[i-1][1]

            # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on"
            data["prev_flag"] = data["flag"].shift()
            data = data.loc[
                ((data["flag"] == "on") & (data["prev_flag"] == "off")) |
                ((data["flag"] == "off") & (data["prev_flag"] == "on")) |
                ((data["flag"] == "off") & (data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
                ((data["flag"] == "on") & (data["prev_flag"].isna()))
            ]

            # Data that we will need:
            last_moment_previous_week = previous_week + pd.Timedelta(hours=23, minutes=59, seconds=59) #We will need the last moment of the week
            first_moment_current_week = (week - pd.Timedelta(days=week.dayofweek)).replace(hour=0, minute=0, second=0) #We will need to the first moment of the current week

            #We have to get the last flag of the previous week
            last_flag_previous_week = previous_week_data.loc[previous_week_data.index[-1], "flag"]

            if last_flag_previous_week == "on":
                # If the last flag from the previous week is "on" then we have to set a new row on the previous week data
                # in the last position to set a flag "off". Then, in the current week data we will add a new row before the
                # first week to set again the alarm to "on"

                # Here we create the new row to set the alarm "off" in the previous week
                new_row_previous_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [last_moment_previous_week],
                    "alarm": ["turn_off_end_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["off"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very end of the week
                new_previous_week_data = pd.concat(
                    [previous_week_data, new_row_previous_week],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i-1][1] = new_previous_week_data

                # Now we have to set the flag "on" in the first moment of the current week:
                # Here we create the new row to set the alarm "on" in the current week
                new_row_current_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [first_moment_current_week],
                    "alarm": ["turn_on_begining_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["on"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very begining of the week
                new_current_week_data = pd.concat(
                    [new_row_current_week, data],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i][1] = new_current_week_data
            else:
                # Simply update the dataframe with the same but with removed rows that contain two identical flags in a row
                grouped_weeks[i][1] = data

        # Once this is done we have to check if the last alarm of the last week is "on". In this case we will add a row turning it off
        # in the last moment of the week:
        last_recorded_week, last_recorded_week_data = grouped_weeks[-1][0], grouped_weeks[-1][1]

        # Get the last flag from the last week
        last_flag = last_recorded_week_data["flag"].values[-1]
        # and get the last moment of the last week
        last_moment_last_recorded_week = last_recorded_week + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if last_flag == "on":
            new_row_last_flag = pd.DataFrame(
                {
                "id": [idd],
                "dated": [last_moment_last_recorded_week],
                "alarm": ["turn_off_end_last_week"], # We will put this in alarm to know witch alarms where inserted by us
                "flag": ["off"],
                "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                }
            )

            last_recorded_week_data = pd.concat(
                [last_recorded_week_data, new_row_last_flag],
                sort=True
            )

            grouped_weeks[-1][1] = last_recorded_week_data

        # Now we have to concat all the dataframes from all the weeks contained in the list grouped_weeks in one big dataframe
        concatenated_weeks = pd.concat(
            [week_data[1] for week_data in grouped_weeks],
            sort=True # Remove the warning of pd.concat
        )

        # At this point there are some cases where we will still have two "on" alarms or two "off" alarms in a row. For example
        # in the case that we have two weeks in a row where we only have "on" alarms utill now the code is going to return the 
        # begining of the end of the previous week with "off", the beggining of the current week with "off" and before the "off"
        # of the end of the week we will still have an "on" alarm. This is caused because the deletion of the same alarms in a row is done 
        # before the add of the new rows in the beggining and end of the week

        # So let's eliminate this cases too:
        # first we have to frop the old prev_flag column that now is useless:
        concatenated_weeks.drop("prev_flag", axis=1, inplace=True)

        # and create the new one:
        concatenated_weeks["prev_flag_concat"] = concatenated_weeks["flag"].shift()

        concatenated_weeks = concatenated_weeks.loc[
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"] == "off")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"] == "on")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"].isna()))
        ]

        # We have some weeks where there is no data and thus if this weeks are between an "on" and "off" flag they will not
        # appear on the dataframe. What we will do is a left join with the variable week generated at the begining
        # of the notebook to see the rows that do not appear. Then we will activate the alarm at the begining of the week
        # and deactivate it at the end of the same week:

        # Set the day to monday
        concatenated_weeks["week"] = concatenated_weeks["dated"].apply(lambda x: return_monday(x))
        # Left join with weeks to detect the missing weeks
        concatenated_weeks_merged = pd.merge(weeks, concatenated_weeks, on="week", how="left")

        # Store the result of the filling:
        filled_dataframe = pd.DataFrame()

        # We add the first row
        if not pd.isna(concatenated_weeks_merged.iloc[0]["alarm"]):
            first_new_row = pd.DataFrame(
                {
                    "week": [concatenated_weeks_merged.iloc[0]["week"]],
                    "alarm": [concatenated_weeks_merged.iloc[0]["alarm"]],
                    "dated": [concatenated_weeks_merged.iloc[0]["dated"]],
                    "flag": [concatenated_weeks_merged.iloc[0]["flag"]],
                    "id": [concatenated_weeks_merged.iloc[0]["id"]]
                }
            )

            filled_dataframe = pd.concat(
                [filled_dataframe, first_new_row],
                sort=True
            )

        # Begin the iteration where we will add one by one the rows to the dataframe filled_dataframe
        for i in range(1, len(concatenated_weeks_merged)-1):

            current_row = concatenated_weeks_merged.iloc[i]

            # If we find a normal row we add it to the dataframe
            if (current_row["alarm"] == "brdpower") | (current_row["alarm"] == "turn_off_end_week") | (current_row["alarm"] == "turn_on_begining_week"):
                current_row["dated"] = pd.to_datetime(current_row["dated"])

                new_row = pd.DataFrame(
                    {
                        "week": [current_row["week"]],
                        "alarm": [current_row["alarm"]],
                        "dated": [current_row["dated"]],
                        "flag": [current_row["flag"]],
                        "id": [current_row["id"]]
                    }
                )
                filled_dataframe = pd.concat([filled_dataframe, new_row], sort=True)

            if not filled_dataframe.empty:
                # If the last row of the filled dataframe is a "turn_off_week" ant the current is a Emty we add the "on" and
                # off for the begining and the end of the week
                last_row_filled_dataframe = filled_dataframe.iloc[-1]

                if (pd.isna(current_row["alarm"])) & (last_row_filled_dataframe["alarm"] in ["turn_off_end_week", "turn_off_end_week_filled"]):

                    new_row_begining_dated = datetime.datetime.combine(current_row["week"], datetime.time(0,0,0))
                    new_row_begining_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_on_begining_week_filled"],
                            "dated": [new_row_begining_dated], # Add the time to the date
                            "flag": ["on"],
                            "id": [idd]
                        }
                    )
                    # We have to add 6 days because the representative of the week in this case is 
                    new_row_end_dated = datetime.datetime.combine(current_row["week"]+ pd.Timedelta(days=6), datetime.time(23,59,59))
                    new_row_end_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_off_end_week_filled"],
                            "dated": [new_row_end_dated], # In this case we add the time to represent the last moment of the week
                            "flag": ["off"],
                            "id": [idd]
                        }
                    )

                    filled_dataframe = pd.concat(
                        [filled_dataframe, new_row_begining_week, new_row_end_week],
                        sort=True
                    )

        # Again we have to do a left join with the dataframe weeks to detect the NaN values:
        filled_dataframe_merged = pd.merge(weeks, filled_dataframe, on="week", how="left")
        
        # Now for each one of the weeks we have to calculate the total time passed between an "on" alarm and an "off" alarm

        # In this dataframe we will store the amount of hours of a ebox that has been malfunctioning for each week
        week_hours_dataframe = pd.DataFrame()
        for week in weeks["week"]:
            # In this variable we will store the amount of hours for this week:
            total_hours = 0
            on_timestamp = None

            # Dataframe with the alarms of the week:
            week_alarms_dataframe = filled_dataframe_merged.loc[filled_dataframe_merged["week"] == week]
            # Iterate trough the df to count the hours:
            for _, row in week_alarms_dataframe.iterrows():
                if row["flag"] == "on":
                    on_timestamp = row["dated"]

                if (row["flag"] == "off") & (on_timestamp is not None):
                    total_hours += (row["dated"] - on_timestamp).total_seconds() / 3600

                    on_timestamp = None

            new_week_hours = pd.DataFrame(
                {
                    "id": [idd],
                    "week": [week],
                    "malfunctioning_hours": [total_hours] 
                }
            )

            week_hours_dataframe = pd.concat(
                [week_hours_dataframe, new_week_hours],
                sort=True
            )

        # Now we want to get the data in the format:
        # row = {"week-4": date_week_prev_4, "hours_week-4": hours_week_prev_4, ..., "week-1": date_week_prev_1, "hours_week-1": hours_week_prev_1, "current_week": date_current_week, "hours_current_week":  "week+1": date_week_next_1, "hours_week+1": hours_week_next_1 ..., "week+4": date_week_next_4, "hours_week+4": hours_week_next_4}

        # In this dataframe we will store the data in the format we have mentioned:
        lag_dataframe = pd.DataFrame()

        # Here we have to follow two different paths. The first one will be when preparing the data for training and the other one will be when 
        # preparing the data for predicting. We shall begin with predicting:

        if for_predicting:
            lag_dataframe = pd.DataFrame(
                {
                    "id": [idd],

                    "week-4": [week_hours_dataframe.iloc[0]["week"]],
                    "hours_week-4": [week_hours_dataframe.iloc[0]["malfunctioning_hours"]],

                    "week-3": [week_hours_dataframe.iloc[1]["week"]],
                    "hours_week-3": [week_hours_dataframe.iloc[1]["malfunctioning_hours"]],

                    "week-2": [week_hours_dataframe.iloc[2]["week"]],
                    "hours_week-2": [week_hours_dataframe.iloc[2]["malfunctioning_hours"]],

                    "week-1": [week_hours_dataframe.iloc[3]["week"]],
                    "hours_week-1": [week_hours_dataframe.iloc[3]["malfunctioning_hours"]],

                    "current_week": [week_hours_dataframe.iloc[4]["week"]],
                    "hours_current_week": [week_hours_dataframe.iloc[4]["malfunctioning_hours"]]
                }
            )

            # Reorder the dataframe so it is in the same order we have defined:
            lag_dataframe = lag_dataframe[columns_order]
            
            # Add the dataframe to the general one:
            general_lag_dataframe = pd.concat(
                [general_lag_dataframe, lag_dataframe],
                sort=True
            )

        else:
            # Begin the loop at 4 and end at -4 so we don't get the error: "Out of range"
            for i in range(4, len(weeks)-4):
                # Create the new row to add:
                to_add_row = pd.DataFrame(
                    {
                        "id": [idd],

                        "week-4": [week_hours_dataframe.iloc[i-4]["week"]],
                        "hours_week-4": [week_hours_dataframe.iloc[i-4]["malfunctioning_hours"]],

                        "week-3": [week_hours_dataframe.iloc[i-3]["week"]],
                        "hours_week-3": [week_hours_dataframe.iloc[i-3]["malfunctioning_hours"]],

                        "week-2": [week_hours_dataframe.iloc[i-2]["week"]],
                        "hours_week-2": [week_hours_dataframe.iloc[i-2]["malfunctioning_hours"]],

                        "week-1": [week_hours_dataframe.iloc[i-1]["week"]],
                        "hours_week-1": [week_hours_dataframe.iloc[i-1]["malfunctioning_hours"]],

                        "current_week": [week_hours_dataframe.iloc[i]["week"]],
                        "hours_current_week": [week_hours_dataframe.iloc[i]["malfunctioning_hours"]],

                        "week+1": [week_hours_dataframe.iloc[i+1]["week"]],
                        "hours_week+1": [week_hours_dataframe.iloc[i+1]["malfunctioning_hours"]],

                        "week+2": [week_hours_dataframe.iloc[i+2]["week"]],
                        "hours_week+2": [week_hours_dataframe.iloc[i+2]["malfunctioning_hours"]],

                        "week+3": [week_hours_dataframe.iloc[i+3]["week"]],
                        "hours_week+3": [week_hours_dataframe.iloc[i+3]["malfunctioning_hours"]],

                        "week+4": [week_hours_dataframe.iloc[i+4]["week"]],
                        "hours_week+4": [week_hours_dataframe.iloc[i+4]["malfunctioning_hours"]]
                    }
                )

                lag_dataframe = pd.concat(
                    [lag_dataframe, to_add_row],
                    sort=True,
                    ignore_index=True
                )

            # Reorder the dataframe so it is in the same order we have defined:
            lag_dataframe = lag_dataframe[columns_order]
            
            # Add the dataframe to the general one:
            general_lag_dataframe = pd.concat(
                [general_lag_dataframe, lag_dataframe],
                sort=True
            )
        
    # Reordenate with the list columns_order:
    general_lag_dataframe = general_lag_dataframe[columns_order]

    end_time = time.time()

    print("Execution time:" + str(end_time - start_time))

    return general_lag_dataframe

# Deprecated. This functions do not have the features for preprocessing the data for predictions

# def big_preprocess_lights(light_alarms: pd.DataFrame) -> pd.DataFrame:
#     """
#     The function converts the data of the alarms into a dataframe that is usable for training the models.
#     Review the coments on the code to understand step by step what the code does.
#     """

#     start_time = time.time()

#     # We create a dataframe of each one of the ids and we will clean the alarms. The final objective is to calculate
#     # the percentage of time the lampost is not working wich is the time that passes from an alarm "on" utill it is 
#     # turned "off". In the dataframe we have cases in wich there are two consecutive "on" or "off" alarms so we have
#     # filter and just keep the first "on" and the first "off" in this case so we get the real time the lampost has been
#     # not functioning. In the case that the alarm is turned "on" during a week and it is not turned "off" untill the next
#     # week we will insert a fake "off" alarm at the last moment of the week and we will turn it "on" again in the first moment
#     # of the following week.

#     # The first step will be to dowload the data:
#     # Read recipe inputs

#     df = light_alarms.copy()

#     # Sort by id and date of the alarm:
#     df["dated"] = pd.to_datetime(df["dated"])
#     df = df.sort_values(["id", "dated"])
#     # As far as we know if an alarm is "set" it means it is "off" so to keep things simple we will sustitute the
#     # "set" values with "off"
#     df.replace(to_replace="set", value="off", inplace=True)

#     # Generate all the weeks from the first day we have data untill the last day. We will need this weeks later to
#     # do a left join with the weeks that there are alarms
#     start_date = df["dated"].min()
#     end_date = return_monday(df["dated"].max()) + pd.Timedelta(days=7)
#     weeks = pd.date_range(start=start_date, end=end_date, freq='W').floor("D").strftime('%Y-%m-%d %H:%M:%S').tolist()
#     # Transform to a dataframe to do operations later:
#     weeks = pd.DataFrame(weeks, columns=["week"])
#     weeks["week"] = pd.to_datetime(weeks["week"])
#     weeks["week"] = weeks["week"].apply(lambda x: return_monday(x))

#     # For each unique id we will implement all the code:
#     # First create the dataframe where we will store all the resturned data:
#     general_lag_dataframe = pd.DataFrame()

#     # List of all the ids:
#     ids_list = df["id"].unique()

#     # Store the name of the columns in oder to later reordenate the final returned dataframe. 
#     columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
#                     "current_week", "hours_current_week", "week+1", "hours_week+1", "week+2", "hours_week+2", "week+3", "hours_week+3", "week+4", "hours_week+4"]

#     # Generate a dataframe for each one of the ids and then apply all the transformations. Once done, add the data
#     # to the dataframe general_lag_dataframe.
#     for idd in ids_list:
#         print(idd)
        
#         # Dataframe of all the alarms for the id "idd":
#         tt = df.loc[df["id"] == idd]
        
#         #For each one of the elements in the list we have: elem[0] contains the week represented by sunday 
#         # and elem[1] has the data in a dataframe
#         grouped_weeks = list(tt.groupby(pd.Grouper(key="dated", freq="W")))
#         # Here we will transform all the tuples to lists so we can change the elements after
#         grouped_weeks = [list(elem) for elem in grouped_weeks]

#         # Now we have to filtrate the empty dataframes that generates groupby for the weeks where there are no alarms
#         grouped_weeks = [elem for elem in grouped_weeks if not elem[1].empty]

#         # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on" for the first element
#         # of the list grouped_weeks. We do it because in the iteration we will not consider it so we have to do it now
#         first_week_data = grouped_weeks[0][1]
#         first_week_data["prev_flag"] = first_week_data["flag"].shift()

#         first_week_data = first_week_data.loc[
#             ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"] == "off")) |
#             ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"] == "on")) |
#             ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
#             ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"].isna()))
#         ]

#         # We store it to the data of the first week:
#         grouped_weeks[0][1] = first_week_data

#         # Here we begin the iteration for all the other weeks:
#         for i, (week, data) in enumerate(grouped_weeks[1:], 1): # We do not consider the first week because it has no previous week to check
#             # Get the data and week from the previous week:
#             previous_week, previous_week_data = grouped_weeks[i-1][0], grouped_weeks[i-1][1]

#             # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on"
#             data["prev_flag"] = data["flag"].shift()
#             data = data.loc[
#                 ((data["flag"] == "on") & (data["prev_flag"] == "off")) |
#                 ((data["flag"] == "off") & (data["prev_flag"] == "on")) |
#                 ((data["flag"] == "off") & (data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
#                 ((data["flag"] == "on") & (data["prev_flag"].isna()))
#             ]

#             # Data that we will need:
#             last_moment_previous_week = previous_week + pd.Timedelta(hours=23, minutes=59, seconds=59) #We will need the last moment of the week
#             first_moment_current_week = (week - pd.Timedelta(days=week.dayofweek)).replace(hour=0, minute=0, second=0) #We will need to the first moment of the current week

#             #We have to get the last flag of the previous week
#             last_flag_previous_week = previous_week_data.loc[previous_week_data.index[-1], "flag"]

#             if last_flag_previous_week == "on":
#                 # If the last flag from the previous week is "on" then we have to set a new row on the previous week data
#                 # in the last position to set a flag "off". Then, in the current week data we will add a new row before the
#                 # first week to set again the alarm to "on"

#                 # Here we create the new row to set the alarm "off" in the previous week
#                 new_row_previous_week = pd.DataFrame(
#                     {
#                     "id": [idd],
#                     "dated": [last_moment_previous_week],
#                     "alarm": ["turn_off_end_week"], # We will put this in alarm to know witch alarms where inserted by us
#                     "flag": ["off"],
#                     "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
#                     }
#                 )

#                 # We create the new df with the new row at the very end of the week
#                 new_previous_week_data = pd.concat(
#                     [previous_week_data, new_row_previous_week],
#                     sort=True # Remove the warning of pd.concat
#                 )
#                 # We update the dataframe in the list grouped_weeks
#                 grouped_weeks[i-1][1] = new_previous_week_data

#                 # Now we have to set the flag "on" in the first moment of the current week:
#                 # Here we create the new row to set the alarm "on" in the current week
#                 new_row_current_week = pd.DataFrame(
#                     {
#                     "id": [idd],
#                     "dated": [first_moment_current_week],
#                     "alarm": ["turn_on_begining_week"], # We will put this in alarm to know witch alarms where inserted by us
#                     "flag": ["on"],
#                     "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
#                     }
#                 )

#                 # We create the new df with the new row at the very begining of the week
#                 new_current_week_data = pd.concat(
#                     [new_row_current_week, data],
#                     sort=True # Remove the warning of pd.concat
#                 )
#                 # We update the dataframe in the list grouped_weeks
#                 grouped_weeks[i][1] = new_current_week_data
#             else:
#                 # Simply update the dataframe with the same but with removed rows that contain two identical flags in a row
#                 grouped_weeks[i][1] = data

#         # Once this is done we have to check if the last alarm of the last week is "on". In this case we will add a row turning it off
#         # in the last moment of the week:
#         last_recorded_week, last_recorded_week_data = grouped_weeks[-1][0], grouped_weeks[-1][1]

#         # Get the last flag from the last week
#         last_flag = last_recorded_week_data["flag"].values[-1]
#         # and get the last moment of the last week
#         last_moment_last_recorded_week = last_recorded_week + pd.Timedelta(hours=23, minutes=59, seconds=59)

#         if last_flag == "on":
#             new_row_last_flag = pd.DataFrame(
#                 {
#                 "id": [idd],
#                 "dated": [last_moment_last_recorded_week],
#                 "alarm": ["turn_off_end_last_week"], # We will put this in alarm to know witch alarms where inserted by us
#                 "flag": ["off"],
#                 "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
#                 }
#             )

#             last_recorded_week_data = pd.concat(
#                 [last_recorded_week_data, new_row_last_flag],
#                 sort=True
#             )

#             grouped_weeks[-1][1] = last_recorded_week_data

#         # Now we have to concat all the dataframes from all the weeks contained in the list grouped_weeks in one big dataframe
#         concatenated_weeks = pd.concat(
#             [week_data[1] for week_data in grouped_weeks],
#             sort=True # Remove the warning of pd.concat
#         )

#         # At this point there are some cases where we will still have two "on" alarms or two "off" alarms in a row. For example
#         # in the case that we have two weeks in a row where we only have "on" alarms utill now the code is going to return the 
#         # begining of the end of the previous week with "off", the beggining of the current week with "off" and before the "off"
#         # of the end of the week we will still have an "on" alarm. This is caused because the deletion of the same alarms in a row is done 
#         # before the add of the new rows in the beggining and end of the week

#         # So let's eliminate this cases too:
#         # first we have to frop the old prev_flag column that now is useless:
#         concatenated_weeks.drop("prev_flag", axis=1, inplace=True)

#         # and create the new one:
#         concatenated_weeks["prev_flag_concat"] = concatenated_weeks["flag"].shift()

#         concatenated_weeks = concatenated_weeks.loc[
#             ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"] == "off")) |
#             ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"] == "on")) |
#             ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
#             ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"].isna()))
#         ]

#         # We have some weeks where there is no data and thus if this weeks are between an "on" and "off" flag they will not
#         # appear on the dataframe. What we will do is a left join with the variable week generated at the begining
#         # of the notebook to see the rows that do not appear. Then we will activate the alarm at the begining of the week
#         # and deactivate it at the end of the same week:

#         # Set the day to monday
#         concatenated_weeks["week"] = concatenated_weeks["dated"].apply(lambda x: return_monday(x))
#         # Left join with weeks to detect the missing weeks
#         concatenated_weeks_merged = pd.merge(weeks, concatenated_weeks, on="week", how="left")

#         # Store the result of the filling:
#         filled_dataframe = pd.DataFrame()

#         # We add the first row
#         if not pd.isna(concatenated_weeks_merged.iloc[0]["alarm"]):
#             first_new_row = pd.DataFrame(
#                 {
#                     "week": [concatenated_weeks_merged.iloc[0]["week"]],
#                     "alarm": [concatenated_weeks_merged.iloc[0]["alarm"]],
#                     "dated": [concatenated_weeks_merged.iloc[0]["dated"]],
#                     "flag": [concatenated_weeks_merged.iloc[0]["flag"]],
#                     "id": [concatenated_weeks_merged.iloc[0]["id"]]
#                 }
#             )

#             filled_dataframe = pd.concat(
#                 [filled_dataframe, first_new_row],
#                 sort=True
#             )

#         # Begin the iteration where we will add one by one the rows to the dataframe filled_dataframe
#         for i in range(1, len(concatenated_weeks_merged)-1):

#             current_row = concatenated_weeks_merged.iloc[i]

#             # If we find a normal row we add it to the dataframe
#             if (current_row["alarm"] == "lightcomm") | (current_row["alarm"] == "lighterr") | (current_row["alarm"] == "turn_off_end_week") | (current_row["alarm"] == "turn_on_begining_week"):
#                 current_row["dated"] = pd.to_datetime(current_row["dated"])

#                 new_row = pd.DataFrame(
#                     {
#                         "week": [current_row["week"]],
#                         "alarm": [current_row["alarm"]],
#                         "dated": [current_row["dated"]],
#                         "flag": [current_row["flag"]],
#                         "id": [current_row["id"]]
#                     }
#                 )
#                 filled_dataframe = pd.concat([filled_dataframe, new_row], sort=True)

#             if not filled_dataframe.empty:
#                 # If the last row of the filled dataframe is a "turn_off_week" ant the current is a Emty we add the "on" and
#                 # off for the begining and the end of the week
#                 last_row_filled_dataframe = filled_dataframe.iloc[-1]

#                 if (pd.isna(current_row["alarm"])) & (last_row_filled_dataframe["alarm"] in ["turn_off_end_week", "turn_off_end_week_filled"]):

#                     new_row_begining_dated = datetime.datetime.combine(current_row["week"], datetime.time(0,0,0))
#                     new_row_begining_week = pd.DataFrame(
#                         {
#                             "week": [current_row["week"]],
#                             "alarm": ["turn_on_begining_week_filled"],
#                             "dated": [new_row_begining_dated], # Add the time to the date
#                             "flag": ["on"],
#                             "id": [idd]
#                         }
#                     )
#                     # We have to add 6 days because the representative of the week in this case is 
#                     new_row_end_dated = datetime.datetime.combine(current_row["week"]+ pd.Timedelta(days=6), datetime.time(23,59,59))
#                     new_row_end_week = pd.DataFrame(
#                         {
#                             "week": [current_row["week"]],
#                             "alarm": ["turn_off_end_week_filled"],
#                             "dated": [new_row_end_dated], # In this case we add the time to represent the last moment of the week
#                             "flag": ["off"],
#                             "id": [idd]
#                         }
#                     )

#                     filled_dataframe = pd.concat(
#                         [filled_dataframe, new_row_begining_week, new_row_end_week],
#                         sort=True
#                     )

#         # Again we have to do a left join with the dataframe weeks to detect the NaN values:
#         filled_dataframe_merged = pd.merge(weeks, filled_dataframe, on="week", how="left")
        
#         # Now for each one of the weeks we have to calculate the total time passed between an "on" alarm and an "off" alarm

#         # In this dataframe we will store the amount of hours of a light that has been malfunctioning for each week
#         week_hours_dataframe = pd.DataFrame()
#         for week in weeks["week"]:
#             # In this variable we will store the amount of hours for this week:
#             total_hours = 0
#             on_timestamp = None

#             # Dataframe with the alarms of the week:
#             week_alarms_dataframe = filled_dataframe_merged.loc[filled_dataframe_merged["week"] == week]
#             # Iterate trough the df to count the hours:
#             for _, row in week_alarms_dataframe.iterrows():
#                 if row["flag"] == "on":
#                     on_timestamp = row["dated"]

#                 if (row["flag"] == "off") & (on_timestamp is not None):
#                     total_hours += (row["dated"] - on_timestamp).total_seconds() / 3600

#                     on_timestamp = None

#             new_week_hours = pd.DataFrame(
#                 {
#                     "id": [idd],
#                     "week": [week],
#                     "malfunctioning_hours": [total_hours] 
#                 }
#             )

#             week_hours_dataframe = pd.concat(
#                 [week_hours_dataframe, new_week_hours],
#                 sort=True
#             )

#         # Now we want to get the data in the format:
#         # row = {"week-4": date_week_prev_4, "hours_week-4": hours_week_prev_4, ..., "week-1": date_week_prev_1, "hours_week-1": hours_week_prev_1, "current_week": date_current_week, "hours_current_week":  "week+1": date_week_next_1, "hours_week+1": hours_week_next_1 ..., "week+4": date_week_next_4, "hours_week+4": hours_week_next_4}

#         # In this dataframe we will store the data in the format we have mentioned:
#         lag_dataframe = pd.DataFrame()

#         # Begin the loop at 4 and end at -4 so we don't get the error: "Out of range"
#         for i in range(4, len(weeks)-4):
#             # Create the new row to add:
#             to_add_row = pd.DataFrame(
#                 {
#                     "id": [idd],

#                     "week-4": [week_hours_dataframe.iloc[i-4]["week"]],
#                     "hours_week-4": [week_hours_dataframe.iloc[i-4]["malfunctioning_hours"]],

#                     "week-3": [week_hours_dataframe.iloc[i-3]["week"]],
#                     "hours_week-3": [week_hours_dataframe.iloc[i-3]["malfunctioning_hours"]],

#                     "week-2": [week_hours_dataframe.iloc[i-2]["week"]],
#                     "hours_week-2": [week_hours_dataframe.iloc[i-2]["malfunctioning_hours"]],

#                     "week-1": [week_hours_dataframe.iloc[i-1]["week"]],
#                     "hours_week-1": [week_hours_dataframe.iloc[i-1]["malfunctioning_hours"]],

#                     "current_week": [week_hours_dataframe.iloc[i]["week"]],
#                     "hours_current_week": [week_hours_dataframe.iloc[i]["malfunctioning_hours"]],

#                     "week+1": [week_hours_dataframe.iloc[i+1]["week"]],
#                     "hours_week+1": [week_hours_dataframe.iloc[i+1]["malfunctioning_hours"]],

#                     "week+2": [week_hours_dataframe.iloc[i+2]["week"]],
#                     "hours_week+2": [week_hours_dataframe.iloc[i+2]["malfunctioning_hours"]],

#                     "week+3": [week_hours_dataframe.iloc[i+3]["week"]],
#                     "hours_week+3": [week_hours_dataframe.iloc[i+3]["malfunctioning_hours"]],

#                     "week+4": [week_hours_dataframe.iloc[i+4]["week"]],
#                     "hours_week+4": [week_hours_dataframe.iloc[i+4]["malfunctioning_hours"]]
#                 }
#             )

#             lag_dataframe = pd.concat(
#                 [lag_dataframe, to_add_row],
#                 sort=True,
#                 ignore_index=True
#             )

#         # Reorder the dataframe so it is in the same order we have defined:
#         lag_dataframe = lag_dataframe[columns_order]
        
#         # Add the dataframe to the general one:
#         general_lag_dataframe = pd.concat(
#             [general_lag_dataframe, lag_dataframe],
#             sort=True
#         )
        
#     # Reordenate with the list columns_order:
#     general_lag_dataframe = general_lag_dataframe[columns_order]

#     end_time = time.time()

#     print("Execution time:" + str(end_time - start_time))

#     return general_lag_dataframe

# def big_preprocess_eboxes(eboxes_alarms: pd.DataFrame) -> pd.DataFrame:
    """
    The function converts the data of the alarms into a dataframe that is usable for training the models.
    Review the coments on the code to understand step by step what the code does.
    """
    
    start_time = time.time()

    # We create a dataframe of each one of the ids and we will clean the alarms. The final objective is to calculate
    # the percentage of time the ebox is not working wich is the time that passes from an alarm "off" utill it is 
    # turned "on". In the dataframe we have cases in wich there are two consecutive "on" or "off" alarms so we have
    # filter and just keep the first "on" and the first "off" in this case so we get the real time the ebox has been
    # not functioning. In the case that the alarm is turned "on" during a week and it is not turned "off" untill the next
    # week we will insert a fake "off" alarm at the last moment of the week and we will turn it "on" again in the first moment
    # of the following week.

    # The first step will be to dowload the data:
    # Read recipe inputs
    df = eboxes_alarms.copy()

    # We will reuse the code of the light alarms for the eboxes: The main thing that we have to modify for the eboxes
    # is that for the alarm subtype brdpower the flag "off" means that the ebox is suffering a breakdown and untill the
    # alarm turns "on" we will consider that the ebox has been down. This is exactly opposite to what happened with the
    # lamposts.

    # The first step is to change the name of the column subtype. Even tough this columns means a subtype of alarm, in this
    # code and from now on we will rename it as alarm so we are able to replicate a similar code as the one used for the lights
    df.rename(columns={"subtype": "alarm"}, inplace=True)

    # There are four different flags in this case: {off, on, offM, onM} depending if the error has been detected digitaly or manualy.
    # we do not care so we fill change this categories to {on off}:
    df["flag"] = df["flag"].replace(["onM"], "on")
    df["flag"] = df["flag"].replace(["offM"], "off")

    # As we already said the flags in the eboxes are opposite as the one in the lamposts, so to be able to use the same
    # code we will just change the values "off" to "on" and the "on" to "off" and then apply the same exact code that
    # we used for the lamposts.
    df["flag"] = df["flag"].replace({'on': 'offf', 'off': 'on'}).replace({"offf": "off"})

    # Sort by id and date of the alarm:
    df["dated"] = pd.to_datetime(df["dated"])
    df = df.sort_values(["id", "dated"])

    # Generate all the weeks from the first day we have data untill the last day. We will need this weeks later to
    # do a left join with the weeks that there are alarms
    start_date = return_monday(df["dated"].min())
    end_date = return_monday(df["dated"].max()) + pd.Timedelta(days=7)
    weeks = pd.date_range(start=start_date, end=end_date, freq='W').floor("D").strftime('%Y-%m-%d %H:%M:%S').tolist()
    # Transform to a dataframe to do operations later:
    weeks = pd.DataFrame(weeks, columns=["week"])
    weeks["week"] = pd.to_datetime(weeks["week"])
    weeks["week"] = weeks["week"].apply(lambda x: return_monday(x))

    # For each unique id we will implement all the code:
    # First create the dataframe where we will store all the resturned data:
    general_lag_dataframe = pd.DataFrame()

    # List of all the ids:
    ids_list = df["id"].unique()

    # Store the name of the columns in oder to later reordenate the final returned dataframe. 
    columns_order = ["id", "week-4", "hours_week-4", "week-3", "hours_week-3", "week-2", "hours_week-2", "week-1", "hours_week-1",
                    "current_week", "hours_current_week", "week+1", "hours_week+1", "week+2", "hours_week+2", "week+3", "hours_week+3", "week+4", "hours_week+4"]

    # Generate a dataframe for each one of the ids and then apply all the transformations. Once done, add the data
    # to the dataframe general_lag_dataframe.
    for idd in ids_list:
        print(idd)
        
        # Dataframe of all the alarms for the id "idd":
        tt = df.loc[df["id"] == idd]
        
        #For each one of the elements in the list we have: elem[0] contains the week represented by sunday 
        # and elem[1] has the data in a dataframe
        grouped_weeks = list(tt.groupby(pd.Grouper(key="dated", freq="W")))
        # Here we will transform all the tuples to lists so we can change the elements after
        grouped_weeks = [list(elem) for elem in grouped_weeks]

        # Now we have to filtrate the empty dataframes that generates groupby for the weeks where there are no alarms
        grouped_weeks = [elem for elem in grouped_weeks if not elem[1].empty]

        # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on" for the first element
        # of the list grouped_weeks. We do it because in the iteration we will not consider it so we have to do it now
        first_week_data = grouped_weeks[0][1]
        first_week_data["prev_flag"] = first_week_data["flag"].shift()

        first_week_data = first_week_data.loc[
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"] == "off")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"] == "on")) |
            ((first_week_data["flag"] == "off") & (first_week_data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((first_week_data["flag"] == "on") & (first_week_data["prev_flag"].isna()))
        ]

        # We store it to the data of the first week:
        grouped_weeks[0][1] = first_week_data

        # Here we begin the iteration for all the other weeks:
        for i, (week, data) in enumerate(grouped_weeks[1:], 1): # We do not consider the first week because it has no previous week to check
            # Get the data and week from the previous week:
            previous_week, previous_week_data = grouped_weeks[i-1][0], grouped_weeks[i-1][1]

            # Eliminate the repeated flags such as "off" followed by "off" or "on" followed by "on"
            data["prev_flag"] = data["flag"].shift()
            data = data.loc[
                ((data["flag"] == "on") & (data["prev_flag"] == "off")) |
                ((data["flag"] == "off") & (data["prev_flag"] == "on")) |
                ((data["flag"] == "off") & (data["prev_flag"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
                ((data["flag"] == "on") & (data["prev_flag"].isna()))
            ]

            # Data that we will need:
            last_moment_previous_week = previous_week + pd.Timedelta(hours=23, minutes=59, seconds=59) #We will need the last moment of the week
            first_moment_current_week = (week - pd.Timedelta(days=week.dayofweek)).replace(hour=0, minute=0, second=0) #We will need to the first moment of the current week

            #We have to get the last flag of the previous week
            last_flag_previous_week = previous_week_data.loc[previous_week_data.index[-1], "flag"]

            if last_flag_previous_week == "on":
                # If the last flag from the previous week is "on" then we have to set a new row on the previous week data
                # in the last position to set a flag "off". Then, in the current week data we will add a new row before the
                # first week to set again the alarm to "on"

                # Here we create the new row to set the alarm "off" in the previous week
                new_row_previous_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [last_moment_previous_week],
                    "alarm": ["turn_off_end_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["off"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very end of the week
                new_previous_week_data = pd.concat(
                    [previous_week_data, new_row_previous_week],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i-1][1] = new_previous_week_data

                # Now we have to set the flag "on" in the first moment of the current week:
                # Here we create the new row to set the alarm "on" in the current week
                new_row_current_week = pd.DataFrame(
                    {
                    "id": [idd],
                    "dated": [first_moment_current_week],
                    "alarm": ["turn_on_begining_week"], # We will put this in alarm to know witch alarms where inserted by us
                    "flag": ["on"],
                    "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                    }
                )

                # We create the new df with the new row at the very begining of the week
                new_current_week_data = pd.concat(
                    [new_row_current_week, data],
                    sort=True # Remove the warning of pd.concat
                )
                # We update the dataframe in the list grouped_weeks
                grouped_weeks[i][1] = new_current_week_data
            else:
                # Simply update the dataframe with the same but with removed rows that contain two identical flags in a row
                grouped_weeks[i][1] = data

        # Once this is done we have to check if the last alarm of the last week is "on". In this case we will add a row turning it off
        # in the last moment of the week:
        last_recorded_week, last_recorded_week_data = grouped_weeks[-1][0], grouped_weeks[-1][1]

        # Get the last flag from the last week
        last_flag = last_recorded_week_data["flag"].values[-1]
        # and get the last moment of the last week
        last_moment_last_recorded_week = last_recorded_week + pd.Timedelta(hours=23, minutes=59, seconds=59)

        if last_flag == "on":
            new_row_last_flag = pd.DataFrame(
                {
                "id": [idd],
                "dated": [last_moment_last_recorded_week],
                "alarm": ["turn_off_end_last_week"], # We will put this in alarm to know witch alarms where inserted by us
                "flag": ["off"],
                "prev_flag": ["Empty"] # Empty because we do not need it anymore and it will help us identify this rows
                }
            )

            last_recorded_week_data = pd.concat(
                [last_recorded_week_data, new_row_last_flag],
                sort=True
            )

            grouped_weeks[-1][1] = last_recorded_week_data

        # Now we have to concat all the dataframes from all the weeks contained in the list grouped_weeks in one big dataframe
        concatenated_weeks = pd.concat(
            [week_data[1] for week_data in grouped_weeks],
            sort=True # Remove the warning of pd.concat
        )

        # At this point there are some cases where we will still have two "on" alarms or two "off" alarms in a row. For example
        # in the case that we have two weeks in a row where we only have "on" alarms utill now the code is going to return the 
        # begining of the end of the previous week with "off", the beggining of the current week with "off" and before the "off"
        # of the end of the week we will still have an "on" alarm. This is caused because the deletion of the same alarms in a row is done 
        # before the add of the new rows in the beggining and end of the week

        # So let's eliminate this cases too:
        # first we have to frop the old prev_flag column that now is useless:
        concatenated_weeks.drop("prev_flag", axis=1, inplace=True)

        # and create the new one:
        concatenated_weeks["prev_flag_concat"] = concatenated_weeks["flag"].shift()

        concatenated_weeks = concatenated_weeks.loc[
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"] == "off")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"] == "on")) |
            ((concatenated_weeks["flag"] == "off") & (concatenated_weeks["prev_flag_concat"].isna())) | # This two last comparisons are for the first rows because they don't have a previous flag
            ((concatenated_weeks["flag"] == "on") & (concatenated_weeks["prev_flag_concat"].isna()))
        ]

        # We have some weeks where there is no data and thus if this weeks are between an "on" and "off" flag they will not
        # appear on the dataframe. What we will do is a left join with the variable week generated at the begining
        # of the notebook to see the rows that do not appear. Then we will activate the alarm at the begining of the week
        # and deactivate it at the end of the same week:

        # Set the day to monday
        concatenated_weeks["week"] = concatenated_weeks["dated"].apply(lambda x: return_monday(x))
        # Left join with weeks to detect the missing weeks
        concatenated_weeks_merged = pd.merge(weeks, concatenated_weeks, on="week", how="left")

        # Store the result of the filling:
        filled_dataframe = pd.DataFrame()

        # We add the first row
        if not pd.isna(concatenated_weeks_merged.iloc[0]["alarm"]):
            first_new_row = pd.DataFrame(
                {
                    "week": [concatenated_weeks_merged.iloc[0]["week"]],
                    "alarm": [concatenated_weeks_merged.iloc[0]["alarm"]],
                    "dated": [concatenated_weeks_merged.iloc[0]["dated"]],
                    "flag": [concatenated_weeks_merged.iloc[0]["flag"]],
                    "id": [concatenated_weeks_merged.iloc[0]["id"]]
                }
            )

            filled_dataframe = pd.concat(
                [filled_dataframe, first_new_row],
                sort=True
            )

        # Begin the iteration where we will add one by one the rows to the dataframe filled_dataframe
        for i in range(1, len(concatenated_weeks_merged)-1):

            current_row = concatenated_weeks_merged.iloc[i]

            # If we find a normal row we add it to the dataframe
            if (current_row["alarm"] == "brdpower") | (current_row["alarm"] == "turn_off_end_week") | (current_row["alarm"] == "turn_on_begining_week"):
                current_row["dated"] = pd.to_datetime(current_row["dated"])

                new_row = pd.DataFrame(
                    {
                        "week": [current_row["week"]],
                        "alarm": [current_row["alarm"]],
                        "dated": [current_row["dated"]],
                        "flag": [current_row["flag"]],
                        "id": [current_row["id"]]
                    }
                )
                filled_dataframe = pd.concat([filled_dataframe, new_row], sort=True)

            if not filled_dataframe.empty:
                # If the last row of the filled dataframe is a "turn_off_week" ant the current is a Emty we add the "on" and
                # off for the begining and the end of the week
                last_row_filled_dataframe = filled_dataframe.iloc[-1]

                if (pd.isna(current_row["alarm"])) & (last_row_filled_dataframe["alarm"] in ["turn_off_end_week", "turn_off_end_week_filled"]):

                    new_row_begining_dated = datetime.datetime.combine(current_row["week"], datetime.time(0,0,0))
                    new_row_begining_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_on_begining_week_filled"],
                            "dated": [new_row_begining_dated], # Add the time to the date
                            "flag": ["on"],
                            "id": [idd]
                        }
                    )
                    # We have to add 6 days because the representative of the week in this case is 
                    new_row_end_dated = datetime.datetime.combine(current_row["week"]+ pd.Timedelta(days=6), datetime.time(23,59,59))
                    new_row_end_week = pd.DataFrame(
                        {
                            "week": [current_row["week"]],
                            "alarm": ["turn_off_end_week_filled"],
                            "dated": [new_row_end_dated], # In this case we add the time to represent the last moment of the week
                            "flag": ["off"],
                            "id": [idd]
                        }
                    )

                    filled_dataframe = pd.concat(
                        [filled_dataframe, new_row_begining_week, new_row_end_week],
                        sort=True
                    )

        # Again we have to do a left join with the dataframe weeks to detect the NaN values:
        filled_dataframe_merged = pd.merge(weeks, filled_dataframe, on="week", how="left")
        
        # Now for each one of the weeks we have to calculate the total time passed between an "on" alarm and an "off" alarm

        # In this dataframe we will store the amount of hours of a ebox that has been malfunctioning for each week
        week_hours_dataframe = pd.DataFrame()
        for week in weeks["week"]:
            # In this variable we will store the amount of hours for this week:
            total_hours = 0
            on_timestamp = None

            # Dataframe with the alarms of the week:
            week_alarms_dataframe = filled_dataframe_merged.loc[filled_dataframe_merged["week"] == week]
            # Iterate trough the df to count the hours:
            for _, row in week_alarms_dataframe.iterrows():
                if row["flag"] == "on":
                    on_timestamp = row["dated"]

                if (row["flag"] == "off") & (on_timestamp is not None):
                    total_hours += (row["dated"] - on_timestamp).total_seconds() / 3600

                    on_timestamp = None

            new_week_hours = pd.DataFrame(
                {
                    "id": [idd],
                    "week": [week],
                    "malfunctioning_hours": [total_hours] 
                }
            )

            week_hours_dataframe = pd.concat(
                [week_hours_dataframe, new_week_hours],
                sort=True
            )

        # Now we want to get the data in the format:
        # row = {"week-4": date_week_prev_4, "hours_week-4": hours_week_prev_4, ..., "week-1": date_week_prev_1, "hours_week-1": hours_week_prev_1, "current_week": date_current_week, "hours_current_week":  "week+1": date_week_next_1, "hours_week+1": hours_week_next_1 ..., "week+4": date_week_next_4, "hours_week+4": hours_week_next_4}

        # In this dataframe we will store the data in the format we have mentioned:
        lag_dataframe = pd.DataFrame()

        # Begin the loop at 4 and end at -4 so we don't get the error: "Out of range"
        for i in range(4, len(weeks)-4):
            # Create the new row to add:
            to_add_row = pd.DataFrame(
                {
                    "id": [idd],

                    "week-4": [week_hours_dataframe.iloc[i-4]["week"]],
                    "hours_week-4": [week_hours_dataframe.iloc[i-4]["malfunctioning_hours"]],

                    "week-3": [week_hours_dataframe.iloc[i-3]["week"]],
                    "hours_week-3": [week_hours_dataframe.iloc[i-3]["malfunctioning_hours"]],

                    "week-2": [week_hours_dataframe.iloc[i-2]["week"]],
                    "hours_week-2": [week_hours_dataframe.iloc[i-2]["malfunctioning_hours"]],

                    "week-1": [week_hours_dataframe.iloc[i-1]["week"]],
                    "hours_week-1": [week_hours_dataframe.iloc[i-1]["malfunctioning_hours"]],

                    "current_week": [week_hours_dataframe.iloc[i]["week"]],
                    "hours_current_week": [week_hours_dataframe.iloc[i]["malfunctioning_hours"]],

                    "week+1": [week_hours_dataframe.iloc[i+1]["week"]],
                    "hours_week+1": [week_hours_dataframe.iloc[i+1]["malfunctioning_hours"]],

                    "week+2": [week_hours_dataframe.iloc[i+2]["week"]],
                    "hours_week+2": [week_hours_dataframe.iloc[i+2]["malfunctioning_hours"]],

                    "week+3": [week_hours_dataframe.iloc[i+3]["week"]],
                    "hours_week+3": [week_hours_dataframe.iloc[i+3]["malfunctioning_hours"]],

                    "week+4": [week_hours_dataframe.iloc[i+4]["week"]],
                    "hours_week+4": [week_hours_dataframe.iloc[i+4]["malfunctioning_hours"]]
                }
            )

            lag_dataframe = pd.concat(
                [lag_dataframe, to_add_row],
                sort=True,
                ignore_index=True
            )

        # Reorder the dataframe so it is in the same order we have defined:
        lag_dataframe = lag_dataframe[columns_order]
        
        # Add the dataframe to the general one:
        general_lag_dataframe = pd.concat(
            [general_lag_dataframe, lag_dataframe],
            sort=True
        )
        
    # Reordenate with the list columns_order:
    general_lag_dataframe = general_lag_dataframe[columns_order]

    end_time = time.time()

    print("Execution time:" + str(end_time - start_time))

    return general_lag_dataframe

