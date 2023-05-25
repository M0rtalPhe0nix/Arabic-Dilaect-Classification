import sqlite3
import pandas as pd
#intializing connector
connector = sqlite3.connect('dialects_database.db')
connector.text_factory = str

# Write a SQL query to retrieve the data
query = '''
SELECT id_text.id, id_text.text,id_dialect.dialect
FROM id_text,id_dialect
where id_text.id = id_dialect.id
'''
#Querying the Data
data = pd.read_sql_query(query, connector)

#checking for Nulls
print(data.isna().sum())

#checking for duplication
print(data.drop(columns=['id']).duplicated().sum())

#Saving Data to CSV

data.to_csv('data.csv', index=False,sep=',', encoding='utf-8-sig', lineterminator='\r\n')


