#scrapy
import scrapy
from scrapy.crawler import Crawler
from scrapy_selenium import SeleniumRequest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.remote_connection import LOGGER as selenium_logger


#utils
import datetime
from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from functools import reduce
import pandas as pd
import time
import argparse
import ast
import os
import urllib3

#debug
from scrapy import cmdline
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
import logging

# Set the project directory
PROJECT_DIR = os.path.dirname(os.path.realpath("__file__"))

# Set Selenium logging level to disable debugging messages
selenium_logger.setLevel(logging.WARNING)

# Set Scrapy logging level to disable debugging messages
logging.getLogger('scrapy').setLevel(logging.WARNING)

# Set the urllib3 logging level to disable debugging messages
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.WARNING)

class WeatherScrapySpider(scrapy.Spider):
    name = 'weather_scrapy'  # Spider name
    allowed_domains = ['wunderground.com']  # Set the allowed domain for scraping

    # Initialize the spider with required arguments
    def __init__(self, mode=None, list_municipalities=None, list_links_wunderground=None, dict_min_max_dates=None, data_dir=None, *args, **kwargs):
        
        super(WeatherScrapySpider, self).__init__(*args, **kwargs)
        
        #Initialize manually passed arguments
        self.mode = mode

        self.list_municipalities = list_municipalities
        self.list_links_wunderground = list_links_wunderground
        self.dict_min_max_dates = dict_min_max_dates

        self.data_dir = data_dir

        #Initialize the logger
        self.setup_logger()

        # Define the webdriver header
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36',
        'Accept-Language': '*',
        'Referer': 'https://www.google.com',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        }

        # Define dict to know if a csv of x municipality has already been created
        self.created_csv = {}
        for municipality in self.list_municipalities:
            self.created_csv[municipality] = False



    def start_requests(self):
        try:
            #Chrome options
            chrome_options = Options()
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--window-size=1280,720")
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0: Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36')
            chrome_options.add_argument('--no_sandbox')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
            chrome_driver_path = os.path.join(PROJECT_DIR, "chromedriver.exe")
            driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver_path)
            
            self.weather_logger.info(f'Chrome Webdriver working..')
        
        except Exception as e:
            self.weather_logger.error(e)



        try:
            #Start requests
            for municipality in self.list_links_wunderground:

                self.weather_logger.info(f'Starting requests for {municipality}')

                # Get the current URL that is being processed by the spider
                current_url = self.list_links_wunderground[municipality]

                # Get the min and max dates for the current municipality from the dict_min_max_dates dictionary
                date_min, date_max = self.dict_min_max_dates[municipality]

                # Convert dates to year and month format
                min_year = int(date_min[0:4])
                max_year = int(date_max[0:4])
                min_month = int(date_min[5:7])
                max_month = int(date_max[5:7])

                # We generate a dictionary with the years and months to scrape for the current municipality
                years_months = {}
                for year in range(min_year, max_year + 1):
                    if min_year == max_year:
                        months = list(range(min_month, max_month +1))
                    elif year == min_year:
                        months = list(range(min_month, 12 + 1))
                    elif year == max_year:
                        months = list(range(1, max_month + 1))
                    else:
                        months = list(range(1,13))
                    years_months[year] = months

                output = pd.DataFrame()

                # Iterate over the months of the period to be scraped
                for year, months in years_months.items():
                    for month in months:
                        # Send the Selenium Request and save in metadata driver, output dataframe, municipality, year and month varibles
                        yield SeleniumRequest(
                            url = f'{current_url}{year}-{month:02d}',
                            wait_time=10,
                            screenshot=True,
                            dont_filter=True,
                            callback=self.parse,
                            meta={'driver':driver, 'municipality': municipality, 'year':year, 'month':month, 'output_df':output},
                            headers=self.headers                        
                        )

 

        except Exception as e:
            self.weather_logger.error(e)


    def parse(self, response):

        try:
            # Retrieve the variables from the metadata
            municipality = response.meta['municipality']
            self.weather_logger.info(f'Extracting weather data for {municipality}')
            year = response.meta['year']
            month = response.meta['month']
            output = response.meta['output_df']

            driver = response.meta['driver']
            driver.get(response.url)


            #Wait for 3 seconds for the page to ensure that all elements of the page are loaded before interacting with them.
            time.sleep(3)
            
            #Extract the HTML source code of the web page 
            r = driver.page_source

            #Use BeautifulSoup to parse the HTML code of the page.
            soup = BS(r, "html.parser")
            container = soup.find('lib-city-history-observation')
            check = container.find('tbody')

            data = []

            #Search for a specific HTML tag in the parsed HTML code that contains the weather data. If the tag is found, 
            #extract the data and stores it in a list called "data".
            for c in check.find_all('tr', class_='ng-star-inserted'):
                for i in c.find_all('td', class_='ng-star-inserted'):
                    trial = i.text
                    trial = trial.strip('  ')
                    data.append(trial)

            #Check the length of the "data" list to determine the number of days and number of variables that where collected
            #Then, we save the different meteorological data in different variables (Temperature, Precipitation, Humidity, etc.)
            if round(len(data) / 17 - 1) == 31:
                Temperature = pd.DataFrame([data[32:128][x:x + 3] for x in range(0, len(data[32:128]), 3)][1:],
                                        columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[128:224][x:x + 3] for x in range(0, len(data[128:224]), 3)][1:],
                                        columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[224:320][x:x + 3] for x in range(0, len(data[224:320]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[320:416][x:x + 3] for x in range(0, len(data[320:416]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[416:512][x:x + 3] for x in range(0, len(data[416:512]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:32][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[512:][1:], columns=['Precipitation'])
                self.weather_logger.info(str(str(year) + "-" + str(month) + ' finished!'))
            elif round(len(data) / 17 - 1) == 28:
                Temperature = pd.DataFrame([data[29:116][x:x + 3] for x in range(0, len(data[29:116]), 3)][1:],
                                        columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[116:203][x:x + 3] for x in range(0, len(data[116:203]), 3)][1:],
                                        columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[203:290][x:x + 3] for x in range(0, len(data[203:290]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[290:377][x:x + 3] for x in range(0, len(data[290:377]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[377:464][x:x + 3] for x in range(0, len(data[377:463]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:29][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[464:][1:], columns=['Precipitation'])
                self.weather_logger.info(str(str(year) + "-" + str(month) + ' finished!'))
            elif round(len(data) / 17 - 1) == 29:
                Temperature = pd.DataFrame([data[30:120][x:x + 3] for x in range(0, len(data[30:120]), 3)][1:],
                                        columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[120:210][x:x + 3] for x in range(0, len(data[120:210]), 3)][1:],
                                        columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[210:300][x:x + 3] for x in range(0, len(data[210:300]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[300:390][x:x + 3] for x in range(0, len(data[300:390]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[390:480][x:x + 3] for x in range(0, len(data[390:480]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:30][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[480:][1:], columns=['Precipitation'])
                self.weather_logger.info(str(str(year) + "-" + str(month) + ' finished!'))
            elif round(len(data) / 17 - 1) == 30:
                Temperature = pd.DataFrame([data[31:124][x:x + 3] for x in range(0, len(data[31:124]), 3)][1:],
                                        columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                Dew_Point = pd.DataFrame([data[124:217][x:x + 3] for x in range(0, len(data[124:217]), 3)][1:],
                                        columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                Humidity = pd.DataFrame([data[217:310][x:x + 3] for x in range(0, len(data[217:310]), 3)][1:],
                                        columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                Wind = pd.DataFrame([data[310:403][x:x + 3] for x in range(0, len(data[310:403]), 3)][1:],
                                    columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                Pressure = pd.DataFrame([data[403:496][x:x + 3] for x in range(0, len(data[403:496]), 3)][1:],
                                        columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                Date = pd.DataFrame(data[:31][1:], columns=data[:1])
                Precipitation = pd.DataFrame(data[496:][1:], columns=['Precipitation'])
                self.weather_logger.info(str(str(year) + "-" + str(month) + ' finished!'))
            else:
                self.weather_logger.warn('Data not in normal length')

            
            #Save the different meteorological data in a compact list 
            dfs = [Date, Temperature, Dew_Point, Humidity, Wind, Pressure, Precipitation]

            #Add exctracted data to a pandas DataFrame
            df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
            
            #Adds the year and month to the date column of DataFrame
            #**Initially, this column only has the day of the data. 
            df_final['Date'] = str(year) + "-" + str(month) + "-" + df_final.iloc[:, :1].astype(str)
            
            # Convert the "Date" column to datetime format
            df_final['Date'] = pd.to_datetime(df_final['Date'])
            
            # Sort the DataFrame by the "Date" column in ascending order
            df_final = df_final.sort_values(by='Date') 

            #Append this Month-Year weather data to the output DataFrame
            output = pd.concat([output, df_final])

            #Wait 10 second to extract the next Year-Month weather data to get rid of being blocked from making requests on the web
            time.sleep(10)

            self.weather_logger.info('Scraper done!')

            #Add column names to the output
            output = output[['Temp_max', 'Temp_avg', 'Temp_min', 'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max',
                            'Hum_avg', 'Hum_min', 'Wind_max', 'Wind_avg', 'Wind_min', 'Pres_max',
                            'Pres_avg', 'Pres_min', 'Precipitation', 'Date']]

            # Save output dataframe
            self.save_output(output, municipality)

        except Exception as e:
            self.weather_logger.error(e)


    def save_output(self, output, municipality):
        try:
            self.weather_logger.info(f'Saving weather data for {municipality}')

            # Get the current directory:
            current_directory = os.getcwd()

            # Get parent directory:
            parent = os.path.split(current_directory)[0]

            # Add the folder to the parent and create the csv file with the meteo data:
            csv_path = parent + "/scrapped_meteo_data/meteo_data.csv"

            if not self.created_csv[municipality]:
                # If this is the first time in this run, create the CSV file
                output.to_csv(csv_path, index=False)
                self.created_csv[municipality] = True
            
            else:
                # If the file was already created in this run, just add the new rows
                output.to_csv(csv_path, mode='a', header=False, index=False)
        
        except Exception as e:
            self.weather_logger.error(e)


    def setup_logger(self):
        ## First block deletes the old logs
        # Get the list of files in the logs directory
        files = os.listdir(self.data_dir)

        # Get the current date to compare with the file dates
        current_date = datetime.datetime.now().date()

        for file in files:
            if file.endswith(".log"):
               # Delete the file
                file_path = os.path.join(self.data_dir, file)
                os.remove(file_path)


        ## Second block creates the actual log
        # Create a logger with the name 'weather_scrapy_logger'
        logger = logging.getLogger('weather_scrapy_logger')
        logger.setLevel(logging.INFO)

        # Generate a unique timestamp to use in the log file name
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a new handler to send messages to the log file
        log_file = os.path.join(self.data_dir, f'weather_scrapy_{current_time}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create a new handler to send messages to the terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter to define the format of messages in the file/console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Save the logger in a class attribute
        self.weather_logger = logger



def run_scrapy(list_municipalities: list, list_links_wunderground: list, dict_min_max_dates: dict[tuple], data_dir: str) -> None:
    os.chdir("./Weather_Spiders")

    # Configura las opciones de Scrapy y crea la instancia del spider
    settings = get_project_settings()
    settings.set("LOG_LEVEL", "WARNING")
    process = CrawlerProcess(settings)

    # Inicia el proceso de Scrapy pasando el nombre del spider y los argumentos
    process.crawl("weather_scrapy", mode='debug', list_municipalities=list_municipalities, list_links_wunderground=list_links_wunderground, dict_min_max_dates=dict_min_max_dates, data_dir=data_dir)

    # Inicia el proceso de Scrapy
    process.start()
    process.stop()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--di", "--meteo_data_folders_directory", help="Provide the local directory on your machine where you want to store the scraped data. Example: /home/leibniz/Desktop/IHMAN/meteo_raw_data", default="/home/leibniz/Desktop/IHMAN/meteo_raw_data")
    parser.add_argument("--mu", "--list_municipalities", help="List of the municipalities the user wants to scrap", nargs='+', default=["illora", "mejorada", "canyelles"])
    parser.add_argument(
        "--dr", 
        "--date_ranges_dict", 
        help = "Pass in dictionary form the range of dates for each one of the municipalities. For example: {'illora': ('2023-04-17', '2023-05-21')}", 
        # default="{'illora': ('2015-01-01', '2023-04-01'), 'mejorada': ('2014-01-01', '2023-04-01'), 'canyelles': ('2015-01-01', '2023-04-01')}"
    )
    parser.add_argument(
        "--li",
        "--scrapy_link",
        help = "List of data directories"       
    )

    args = parser.parse_args()
    municipalities_list = args.mu
    dict_min_max_dates = ast.literal_eval(args.dr)
    links = ast.literal_eval(args.li)
    data_dir = args.di

    run_scrapy(
        list_municipalities = municipalities_list,
        list_links_wunderground = links,
        dict_min_max_dates = dict_min_max_dates,
        data_dir = data_dir
    )