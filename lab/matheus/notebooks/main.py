import funcs.data_wrangling as dw

# data = dw.ons_data(freq='h', ano_inicio=2000, ano_fim=2023, idreg="S")
# data.read()
# data.check_date_column(printer=True)
# data.insert_missing_dates()
# data.get_data_description(plot=False)
# data.fill_na(_method="linear")
# df = data.data_treated
# print(df.info())

data = dw.ons_data(freq='h', ano_inicio=2000, ano_fim=2023, idreg="S")
dw.pipeline(data)