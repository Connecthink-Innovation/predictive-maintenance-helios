import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import pickle

def calculate_accuracies(y: pd.DataFrame) -> dict:
    """
    Takes as input a dataframe with two columns. The first one with the real values and
    the second one with the predictions and returns the accuracies of the predictions
    """

    total_accuracy = len(y.loc[y["error_next_four_weeks"] == y["predictions"]]) / float(len(y))
    # Percentage of Yes detected
    yes_accuracy = len(y.loc[(y["error_next_four_weeks"] == "Yes") & (y["predictions"] == "Yes")]) / float(len(y.loc[(y["error_next_four_weeks"] == "Yes")]))
    # Same with No:
    if len(y.loc[(y["error_next_four_weeks"] == "No")]) != 0: # In case we get a zero division
        no_accuracy = len(y.loc[(y["error_next_four_weeks"] == "No") & (y["predictions"] == "No")]) / float(len(y.loc[(y["error_next_four_weeks"] == "No")]))
    else:
        no_accuracy = None
    return {"total_accuracy": total_accuracy, "yes_accuracy": yes_accuracy, "no_accuracy": no_accuracy}

def adboc_predictor(row, ada_model, ada_sudden_model, prob_threshold_ada_model, prob_threshold_ada_sudden_model, return_probs=False):
    # Ada Boost Combined predictor
    # Gets two models and a row in the form of a pandas series and returns the prediction for that row 
    # using the models

    # Get the number of past weeks with errors
    total_week_errors = (
        row["hours_current_week"] + 
        row["hours_week-1"] + 
        row["hours_week-2"] + 
        row["hours_week-3"] +
        row["hours_week-4"]
    ).iloc[0]
    
    # This "if" checks if we want to return probabilities or we want the clasification. In the case of checking the
    # accuracies during training we will use the default value for return_probs. For returning predictions we 
    # are interested in getting probabilities to return to the user, so in this case we will set return_probs = True
    if return_probs:
        if total_week_errors <= 2:
            yes_prob = ada_sudden_model.predict_proba(row)[:, 1][0]
            return (yes_prob, "ada_sudden_model")
        if total_week_errors > 2:
            yes_prob = ada_model.predict_proba(row)[:, 1][0]
            return (yes_prob, "ada_model")
    else:
        if total_week_errors <= 2:
            yes_prob = ada_sudden_model.predict_proba(row)[:, 1]
            if yes_prob >= prob_threshold_ada_sudden_model:
                return "Yes"
            else:
                return "No"

        if total_week_errors > 2:
            yes_prob = ada_model.predict_proba(row)[:, 1]
            if yes_prob >= prob_threshold_ada_model:
                return "Yes"
            else:
                return "No"

def test_adboc_model(x, y, ada_model, ada_sudden_model, prob_threshold_ada_model, prob_threshold_ada_sudden_model):
    yy = y.copy()
    
    # Let's make the final predictions:
    predictions = []
    for _, row in x.iterrows():
        
        predictions.append(
            adboc_predictor(
                pd.DataFrame([row]), 
                ada_model, 
                ada_sudden_model, 
                prob_threshold_ada_model, 
                prob_threshold_ada_sudden_model
            )
        )

    yy["predictions"] = predictions
    
    return (calculate_accuracies(yy), yy)

def adboc_predict(df: pd.DataFrame, ada_model: AdaBoostClassifier, ada_sudden_model: AdaBoostClassifier, prob_threshold_ada_model: float, prob_threshold_ada_sudden_model: float) -> dict:

    predictions = {}

    for _, row in df.iterrows():

        predictions[row["id"]] = adboc_predictor(
            row = pd.DataFrame([row]).drop("id", axis=1),
            ada_model = ada_model,
            ada_sudden_model = ada_sudden_model,
            prob_threshold_ada_model = prob_threshold_ada_model,
            prob_threshold_ada_sudden_model = prob_threshold_ada_sudden_model,
            return_probs = True
        )
    
    return predictions

def make_predictions_lights(light_alarms: pd.DataFrame, model_type: str = "default") -> None:

    # In this case we don't use the ada boost combined predictor and the model will take into accout the readings. This
    # model has an overall better accuracy than the adboc but fails to detect the sudden errors
    if model_type == "default":
        with open("predictive_models/ada_model_readings.pk1", "rb") as file:
            ada_model = pickle.load(file)
        with open("predictive_models/ada_prob_readings.pk1", "rb") as file:
            prob_ada_model = pickle.load(file)["prob_ada_model"]
        
        df = light_alarms.copy()

        # Drop some usless columns for the model:
        drop_cols = [
            col for col in df.columns if
                (col in ["lat", "lon"]) | 
                (col == "Unnamed: 0") |
                (col.startswith("week")) |
                (col == "current_week") |
                (col == "type") |
                (col == "ebox_id") |
                (col == "location")
        ]
        df = df.drop(drop_cols, axis=1)

        # Interpolate some left missing values:
        df = df.fillna(df.mean(numeric_only=True))

        predictions = ada_model.predict_proba(df.drop("id", axis=1))[:, 1]

        predictions_out = pd.DataFrame(
            {
                "id": df["id"],
                "pred": predictions
            }
        )

        print("Probability threshold recommended for this model for lights: " + str(prob_ada_model))
        print(predictions_out)

    # In this case we will use the adboc model. This model has a worse overall accuracy than the default but has a better chance 
    # at dettecting sudden errors. The model does not use the readings in this case.
    if model_type == "adboc":
        with open("predictive_models/ada_model.pk1", "rb") as file:
            ada_model = pickle.load(file)
        with open("predictive_models/ada_sudden_model.pk1", "rb") as file:
            ada_sudden_model = pickle.load(file)
        with open("predictive_models/ada_prob.pk1", "rb") as file:
            probs_dict = pickle.load(file)
            prob_ada_model = probs_dict["prob_ada_model"]
            prob_sudden_model = probs_dict["prob_sudden_model"]
        
        df = light_alarms.copy()
        drop_cols = [
                    col for col in df.columns if 
                    (col.startswith("power")) | (col.startswith("Active")) | (col.startswith("Reactive") | 
                    (col in ["lat", "lon"])) | 
                    (col == "Unnamed: 0") |
                    (col.startswith("week")) |
                    (col == "current_week") |
                    (col == "type") |
                    (col == "ebox_id") |
                    (col == "location")
                ]
        df = df.drop(drop_cols, axis=1)
        df = df.fillna(df.mean(numeric_only=True))

        predictions = adboc_predict(
            df = df,
            ada_model = ada_model,
            ada_sudden_model = ada_sudden_model,
            prob_threshold_ada_model = prob_ada_model,
            prob_threshold_ada_sudden_model = prob_sudden_model,
        )

        predictions_out = pd.DataFrame(
            {
                "id": predictions.keys(),
                "pred": predictions.values()
            }
        )

        print("Probability threshold recommended for the model ada_model for lights: " + str(prob_ada_model))
        print("Probability threshold recommended for the model ada_sudden_model for lights: " + str(prob_sudden_model))
        print(predictions_out)

def make_predictions_eboxes(eboxes_alarms: pd.DataFrame) -> None:

    with open("predictive_models/ada_model_eboxes.pk1", "rb") as file:
        ada_model = pickle.load(file)

    with open("predictive_models/ada_prob_eboxes.pk1", "rb") as file:
        probs_dict = pickle.load(file)
        prob_ada_model = probs_dict["prob_ada_model"]
    
    df = eboxes_alarms.copy()
    drop_cols = [
                col for col in df.columns if 
                (col.startswith("power")) | (col.startswith("Active")) | (col.startswith("Reactive") | 
                (col in ["lat", "lon"])) | 
                (col == "Unnamed: 0") |
                (col.startswith("week")) |
                (col == "current_week") |
                (col == "type") |
                (col == "location")
            ]
    df = df.drop(drop_cols, axis=1)
    df = df.fillna(df.mean(numeric_only=True))

    predictions = ada_model.predict_proba(df.drop("id", axis=1))[:, 1]

    predictions_out = pd.DataFrame(
        {
            "id": df["id"],
            "pred": predictions
        }
    )

    print("Probability threshold recommended for this model for eboxes: " + str(prob_ada_model))
    print(predictions_out)