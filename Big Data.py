import pandas as pd
import mysql.connector

data = mysql.connector.connect(
    host='localhost',
    user='root',
    password='@MySeniorProJecT21',
    database='big_data'
)

cursor = data.cursor()
csv_file = "realdonaldtrump.csv"
df = pd.read_csv(csv_file)
print(df)

# Insert data into user_data table
#for _, row in df[['UserScreenName', 'UserName']].drop_duplicates().iterrows():
#    cursor.execute("INSERT INTO user_data (full_name, username) VALUES (%s, %s)", tuple(row))

# Insert data into Tweets table
for _, row in df[['id', 'content']].iterrows():
    cursor.execute("INSERT INTO Tweets (tweet_id, tweet_text) VALUES (%s, %s)", tuple(row))

# Commit changes and close cursor
data.commit()
cursor.close()

# Close connection
data.close()