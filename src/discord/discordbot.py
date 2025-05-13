from typing import List, Callable, Awaitable

import discord
from discord import Message

class DiscordBot:
    token: str
    client: discord.Client

    on_message_handlers = List[Callable[[str], Awaitable[None]]]

    def __init__(self, token, intents: discord.Intents):
        self.on_message_handlers = []
        self.token = token
        self.client = discord.Client(intents=intents)


    def run(self):
        @self.client.event
        async def on_ready():
            print(f'Logged in as {self.client.user} (ID: {self.client.user.id})')

        @self.client.event
        async def on_message(message: Message):
            if message.author == self.client.user:
                return

            for handler in self.on_message_handlers:
                await handler(message)

        self.client.run(self.token)


    async def send_reply(self, msg_channel, message: str):
        if msg_channel.author == self.client.user:
            return

        await msg_channel.channel.send(message)


    def add_message_handler(self, handler: List[Callable[[str], Awaitable[None]]]):
        if callable(handler):
            self.on_message_handlers.append(handler)