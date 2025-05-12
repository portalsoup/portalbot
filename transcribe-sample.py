import discord
from discord.ext import commands
import whisper
import asyncio
import os

# Requirements:
# pip install py-cord whisper openai-whisper PyNaCl
# Ensure ffmpeg is installed and in PATH

# Load Whisper model once
MODEL = whisper.load_model("base")

# Create a bot with message intent disabled (not needed for voice)
intents = discord.Intents.default()
intents.voice_states = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Directory to store temporary audio files
os.makedirs("recordings", exist_ok=True)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("Ready to record voice channels.")

@bot.command(name='join')
async def join(ctx):
    """Bot joins the voice channel of the command issuer."""
    if ctx.author.voice and ctx.author.voice.channel:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined **{channel.name}**!")
    else:
        await ctx.send("You are not in a voice channel.")

@bot.command(name='record')
async def start_record(ctx):
    """Start recording per-user audio in the current channel."""
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return await ctx.send("Bot is not in a voice channel. Use !join.")

    # Use WaveSink to record WAV per user
    sink = discord.sinks.WaveSink()
    vc.start_recording(
        sink,
        callback=record_finished,
        ctx=ctx  # Pass context through to callback
    )
    await ctx.send("Recording started! Use !stop to finish.")

@bot.command(name='stop')
async def stop_record(ctx):
    """Stop recording and process audio streams."""
    vc = ctx.voice_client
    if not vc or not vc.is_recording():
        return await ctx.send("No recording in progress.")

    vc.stop_recording()
    await ctx.send("Recording stopped. Processing transcripts...")

async def record_finished(sink: discord.sinks.Sink, ctx: commands.Context):
    """Callback executed after stop_recording. Transcribes each user's audio."""
    # sink.audio_data: dict[user_id] -> discord.sinks.AudioData
    for user_id, audio_data in sink.audio_data.items():
        # Save file from AudioData
        filename = f"recordings/{user_id}.wav"
        audio_data.file.save(filename)

        # Transcribe with Whisper
        result = MODEL.transcribe(filename)
        text = result.get("text", "[no speech detected]")

        # Fetch user's display name
        user = await bot.fetch_user(user_id)
        name = user.display_name if hasattr(user, 'display_name') else str(user)

        # Send transcript
        await ctx.send(f"**{name}:** {text.strip()}")

    # Cleanup files
    for f in os.listdir("recordings"):
        os.remove(os.path.join("recordings", f))

# Run the bot
if __name__ == '__main__':
    TOKEN = os.getenv('DISCORD_TOKEN')  # Set your bot token as an environment variable
    if not TOKEN:
        print("Error: DISCORD_TOKEN env var not set.")
        exit(1)
    bot.run(TOKEN)
