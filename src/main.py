
import utils.data_wrangling as dw
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\user\Projetos\load_forecasting\src")

print(os.getcwd())

data = dw.inmet_data(2012, 2023)
# df = data.read_parquet()
# print(df.info())
# print(df.shape)
# print(df.isna().sum())
#print(os.listdir())
#data.download()
#data.build_database()
#df = data.read_parquet()
#data.check_date_column()

#print(df.head())
#print(df.tail())

temp_data = dw.inmet_data(2012,2023)
temp_data.read_parquet()
df = temp_data.aggregate_by_hour()
print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())
print(df.dtypes)