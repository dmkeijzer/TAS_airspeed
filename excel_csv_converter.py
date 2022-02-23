import pandas as pd
import os

files = os.listdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")
os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")

print("Process has started")

# read_file = pd.read_excel(files[0], sheet_name= "Untitled")

print(f"{files[0]} has been loaded in")

clean_path = r'C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data'
clean_path += files[0]
clean_path =  clean_path.replace(".xlsx", ".csv")

print(f"test = {clean_path}")
read_file.to_csv (r'C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data', index = None, header=True)

print(f"{files[0]} has been converted to csv and put in clean data")

# read_file = pd.read_excel (r'Path where the Excel file is stored\File name.xlsx', sheet_name='Your Excel sheet name')
# read_file.to_csv (r'Path to store the CSV file\File name.csv', index = None, header=True)

