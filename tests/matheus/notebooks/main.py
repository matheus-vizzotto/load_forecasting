import funcs.data_wrangling as dw

# # data = dw.ons_data(freq='h', ano_inicio=2000, ano_fim=2023, idreg="S")
# # data.read()
# # data.check_date_column(printer=True)
# # data.insert_missing_dates()
# # data.get_data_description(plot=False)
# # data.fill_na(_method="linear")
# # df = data.data_treated
# # print(df.info())

# data1 = dw.ons_data(freq='d', ano_inicio=2000, ano_fim=2023, idreg="N")
# data1.read()
# df1 = data1.data
# data2 = dw.ons_data(freq='d', ano_inicio=2000, ano_fim=2023, idreg="N")
# df2 = dw.pipeline(data1)
# print(df1.info())
# print(df2.info())
# #print("\nDados antes:", df2.info())
# #print("Dados depois:", df1.info())


import funcs.data_wrangling as dw
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\user\Projetos\load_forecasting\tests\matheus\notebooks")

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