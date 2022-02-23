import pandas as pd
import os

files = os.listdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")
os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")

print("Process has started")

read_file = pd.read_excel(files[0], sheet_name= "Untitled")

print(f"{files[0]} has been loaded in")

clean_path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data\cleandata_"
clean_path += files[0]
clean_path =  clean_path.replace(".xlsx", ".csv")

print(f"test = {clean_path}")
read_file.to_csv (clean_path, index = None, header=True)

print(f"{clean_path} has been converted to csv and put in clean data")



