import funcs.data_wrangling as dw

# data = dw.ons_data(freq='h', ano_inicio=2000, ano_fim=2023, idreg="S")
# data.read()
# data.check_date_column(printer=True)
# data.insert_missing_dates()
# data.get_data_description(plot=False)
# data.fill_na(_method="linear")
# df = data.data_treated
# print(df.info())

data1 = dw.ons_data(freq='d', ano_inicio=2000, ano_fim=2023, idreg="N")
data1.read()
df1 = data1.data
data2 = dw.ons_data(freq='d', ano_inicio=2000, ano_fim=2023, idreg="N")
df2 = dw.pipeline(data1)
print(df1.info())
print(df2.info())
#print("\nDados antes:", df2.info())
#print("Dados depois:", df1.info())