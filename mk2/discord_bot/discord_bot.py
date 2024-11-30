import discord
import requests
import os
from dotenv import load_dotenv

intents = discord.Intents.default()
client = discord.Client(intents=intents)
BACKEND_URL = "http://fastapi_backend:8000/generate"
load_dotenv(dotenv_path="discord.env")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!generate"):
        recipient_name = message.content[len("!generate "):].strip()

        if not recipient_name:
            await message.channel.send("Please provide the recipient's name.")
            return

        prompt = f"Write a love letter to {recipient_name}."

        payload = {
            "model": "llava:13b",  # Make sure this matches your model name
            "prompt": prompt,
            "images": []
        }

        try:
            response = requests.post(BACKEND_URL, json=payload)
            response.raise_for_status()
            response_data = response.json()

            if "text" in response_data:
                await message.channel.send(response_data["text"])
            else:
                await message.channel.send("Sorry, no text generated.")
        except requests.exceptions.RequestException as e:
            await message.channel.send(f"Error communicating with the backend: {e}")

client.run(DISCORD_TOKEN)
