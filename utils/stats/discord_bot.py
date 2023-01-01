#NOT IN USE
import discord
import hashlib
import sqlite3

BOT_TOKEN = ""
PATH_TO_USER_DB = "database.db"

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    # Check if the message was sent in a DM
    if isinstance(message.channel, discord.DMChannel):
        if message.content.startswith('!register'):
            # Split the message content into a list of arguments
            args = message.content.split()

            # Check that the correct number of arguments has been provided (username and password)
            if len(args) != 3:
                await message.channel.send('Invalid number of arguments. Use the following format: !register <username> <password>')
                return

            # Extract the username and password from the arguments
            username = args[1]
            password = args[2]

            # Hash the password using the SHA256 algorithm
            hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()

            # Connect to the database
            conn = sqlite3.connect(PATH_TO_USER_DB)
            cursor = conn.cursor()

            # Check if the user already has an account registered
            cursor.execute('''SELECT * FROM users WHERE user_id=?''', (message.author.id,))
            result = cursor.fetchone()
            if result is not None:
                await message.channel.send('You already have an account registered')
                return

            # Check if there is already a user with the same username
            cursor.execute('''SELECT * FROM users WHERE username=?''', (username,))
            result = cursor.fetchone()
            if result is not None:
                await message.channel.send('There is already a user with that username')
                return

            # Insert the user into the database
            cursor.execute('''INSERT INTO users(user_id, username, password) VALUES(?, ?, ?)''', (message.author.id, username, hashed_password))
            conn.commit()
            await message.channel.send(f'Successfully registered user {username}')


client.run(BOT_TOKEN)
