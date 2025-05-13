import argparse
from mailbox import Message

import discord

from src.ai.AIPipeline import AIPipeline, Response
from src.ai.AIPipelineContext import AIPipelineContext
from src.discord.discordbot import DiscordBot
from src.globals import init_cuda, load_prompt

class App:
    app_context: AIPipelineContext
    pipeline: AIPipeline
    messages: list[dict[str, str]]

    discord_bot: DiscordBot

    def __init__(self, args: argparse.Namespace):
        if not args.no_cuda:
            init_cuda()

        self.app_context = AIPipelineContext(args.model, verbose=args.verbose, think=args.think)
        self.pipeline = AIPipeline(self.app_context)
        self.messages = load_prompt(args.prompt)

        intents = discord.Intents.default()
        intents.message_content = True
        self.discord_bot = DiscordBot(args.discord_token, intents)

    def main(self):
        self.discord_bot.add_message_handler(self.incoming_discord_message)

        # Start bot
        self.discord_bot.run()

    async def incoming_discord_message(self, message: Message):
        self._append_message(message.author.global_name, message.content)
        response = self._generate_response()
        print(self.messages)
        print("Got a response:")
        print(response.response)
        await self.discord_bot.send_reply(message, response.response)

    def _append_message(self, username: str, content: str):
        self.messages.append({"role": "user", "content": f"[${username}] ${content}"})

    def _print_response(self, response: Response):
        if self.app_context.think:
            print(f"${response.thinking.strip()}\n\nResponse:\n")

        print(response.response.strip())

    def _generate_response(self) -> Response:
        response = self.pipeline.query(self.messages)
        self._append_message("assistant", response.response)
        return response