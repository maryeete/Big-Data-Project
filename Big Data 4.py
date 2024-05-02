import pandas as pd
import mysql.connector

data = mysql.connector.connect(
    host='localhost',
    user='root',
    password='@MySeniorProJecT21',
    database='big_data'
)

cursor = data.cursor()
csv_file = "Event Category.csv"
df = pd.read_csv(csv_file)
print(df)
print(df.dtypes)

df_subset = df.head(150)

# Insert data into user_data table
#for _, row in df[['UserScreenName', 'UserName']].drop_duplicates().iterrows():
#    cursor.execute("INSERT INTO user_data (full_name, username) VALUES (%s, %s)", tuple(row))

# Insert data into Tweets table
# Update data in Tweets table
for index, row in df_subset.iterrows():
    event_label = row['event_label']
    tweet_id = row['tweet_id']
    cursor.execute("UPDATE Tweets SET event_label = %s WHERE tweet_id = %s", (event_label, tweet_id))

# Commit changes and close cursor
data.commit()
cursor.close()

# Close connection
data.close()