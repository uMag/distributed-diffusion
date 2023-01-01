#NOT IN USE
import sqlite3

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create the data table
cursor.execute('''
CREATE TABLE data (
    username TEXT NOT NULL,
    machine_uuid TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data TEXT NOT NULL
)
''')

# Save the changes
conn.commit()

# Close the connection
conn.close()
