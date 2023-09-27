import datetime

def return_monday(my_datetime: datetime) -> datetime:
    """
    Returns the date of the monday of the week this function will appear many times in the code
    """
    return my_datetime.date() - datetime.timedelta(days=my_datetime.weekday())

