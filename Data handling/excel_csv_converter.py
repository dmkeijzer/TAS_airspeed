import pandas as pd
import os
import time

files = os.listdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")
os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\OG_data\data\microphone_data")

t1 = time.time()

#reading loop

for counter, file in enumerate(files, start= 1):

    print(f"\n file {counter}")
    print(f"\n{file} is being loaded into memory")

    read_file = pd.read_excel(file, sheet_name= "Untitled")

    print(f"{file} has been loaded into memory")

    clean_path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data\cleandata_"
    clean_path += file
    clean_path =  clean_path.replace(".xlsx", ".csv")

    read_file.to_csv (clean_path, index = None, header=True)

    print(f"{file} has been converted to csv and saved")

t2 = time.time()

print(f"execution time = {t2-t1} [s] or {(t2 -t1) / 60} [minutes]")
    


