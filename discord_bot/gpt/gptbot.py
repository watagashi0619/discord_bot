import math
import os
import tomllib
import traceback
from logging import config, getLogger
from typing import List

import discord
import openai
from discord import app_commands
from dotenv import load_dotenv

current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
log_folder_abspath = os.path.join(grandparent_folder_abspath, "logs")
configpath = os.path.join(grandparent_folder_abspath, "pyproject.toml")
basename = os.path.basename(__file__).split(".")[0]
with open(configpath, "rb") as f:
    log_conf = tomllib.load(f).get("logging")
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(log_folder_abspath, f"{basename}.log")

print(log_conf)

logger = getLogger(__name__)
config.dictConfig(log_conf)
dotenvpath = os.path.join(grandparent_folder_abspath, ".env")
load_dotenv(dotenvpath)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN_GPT")
openai.api_key = os.getenv("OPENAI_API_KEY")
BOT_NAME = "chat-gpt"
CHANNEL_ID = os.getenv("CHANNEL_ID_GPT")
model_engine = "gpt-3.5-turbo"
chat_log = [
    {
        "role": "system",
        "content": "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
        + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
        + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。",
    }
]
total_token = 0


def round_to_digits(val: float, digits: int) -> float:
    """
    Rounds the given value to the specified number of digits.

    Args:
        val (float): The value to be rounded.
        digits (int): Number of digits to round to. Must be a positive integer.

    Returns:
        float: The value rounded to the specified number of digits.

    Examples:
        >>> round_to_digits(3.14159, 2)
        3.1
        >>> round_to_digits(0.00123456, 5)
        0.0012346
    """
    if val == 0:
        return 0
    else:
        return round(val, -int(math.floor(math.log10(abs(val))) + (1 - digits)))


def split_string(text: str) -> List[str]:
    """
    Split a long string, possibly containing newlines and code blocks, into a
    list of strings each with maximum length 2000.

    The split is performed at the last newline before the 2000 character limit
    is reached, or at the 2000th character if the string is in a code block.
    If a split occurs within a code block, appropriate code block tags are
    added to maintain correct formatting.

    Empty strings are removed from the final list.

    Args:
        text (str): The string to split.

    Returns:
        List[str]: The list of split strings.
    """
    ret_list = []
    buffer = ""
    code_block_flag = False
    for line in text.split("\n"):
        if "```" in line:
            code_block_flag = not code_block_flag
        if len(buffer + line + "\n") <= 2000 or (len(buffer + line + "\n") > 2000 and code_block_flag):
            buffer += line + "\n"
            if code_block_flag and len(buffer) > 2000:
                ret_list.append(buffer)
                buffer = "```\n"
        else:
            ret_list.append(buffer)
            buffer = line + "\n"
    if buffer:
        ret_list.append(buffer)

    ret_list_clean = [s for s in ret_list if s != ""]
    return ret_list_clean


@client.event
async def on_ready():
    """on ready"""
    print(f"We have logged in as {client.user}")
    await tree.sync()


@tree.command(name="gpt", description="chat gptを呼び出す")
async def gpt(interaction: discord.Interaction, prompt: str):
    """call ChatGPT bot.

    Args:
        interaction (discord.Interaction): interaciton.
        prompt (str): message to be passed on to ChatGPT.
    """
    logger.info("command: gpt")
    global chat_log
    global model_engine
    global total_token
    # msg = await message.reply("生成中...", mention_author=False)
    await interaction.response.defer()
    if not prompt:
        await interaction.followup.send("質問内容がありません")
        return
    chat_log.append({"role": "user", "content": prompt})
    logger.info("user: " + prompt)
    retries = 3
    while retries > 0:
        try:
            completion = openai.ChatCompletion.create(
                model=model_engine,
                messages=chat_log,
            )
            response = "> " + prompt + "\n\n"
            response += completion["choices"][0]["message"]["content"]
            chat_log.append(completion["choices"][0]["message"].to_dict())
            logger.info("assistant: " + completion["choices"][0]["message"].to_dict()["content"])
            if model_engine == "gpt-3.5-turbo":
                price = round_to_digits(
                    completion["usage"]["prompt_tokens"] * 0.0015 / 1000
                    + completion["usage"]["completion_tokens"] * 0.002 / 1000,
                    3,
                )
            elif model_engine == "gpt-4":
                price = round_to_digits(
                    completion["usage"]["prompt_tokens"] * 0.03 / 1000
                    + completion["usage"]["completion_tokens"] * 0.06 / 1000,
                    3,
                )
                response = response + f"\n(USAGE: {price} USD)"
            total_token += completion["usage"]["total_tokens"]
            if total_token > 4096:
                chat_log = chat_log[:1] + chat_log[2:]
            await interaction.followup.send(response)
            break
        except openai.error.InvalidRequestError as e:
            retries -= 1
            if retries > 1:
                logger.exception(e)
                await interaction.followup.send(
                    f"OpenAI APIでエラーが発生しました。リトライします（残回数{retries}）。\n{traceback.format_exception_only(e)}"
                )
                if "Please reduce the length of the messages." in traceback.format_exception_only(e):
                    chat_log = chat_log[:1] + chat_log[2:]
            else:
                logger.exception(e)
                await interaction.followup.send(f"OpenAI APIでエラーが発生しました。\n{traceback.format_exception_only(e)}")
        except discord.errors.HTTPException as e:
            logger.exception(e)
            await interaction.followup.send(f"Discord APIでエラーが発生しました。\n{traceback.format_exception_only(e)}")
            break
        except Exception as e:
            logger.exception(e)
            await interaction.followup.send(f"エラーが発生しました。\n{traceback.format_exception_only(e)}")
            break


@tree.command(name="gpt-hflush", description="chat gptのチャット履歴を消去する")
async def gpt_delete(interaction: discord.Interaction):
    """delete chat history with ChatGPT.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-hflush")
    global chat_log
    chat_log = [
        {
            "role": "system",
            "content": "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
            + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
            + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。",
        }
    ]
    logger.info("Deleted chat logs.")
    response = "チャット履歴を削除しました。"
    await interaction.response.send_message(response)


@tree.command(name="gpt-switch", description="chat gptモデルをgpt-3.5-turboとgpt-4の間で切り替える")
async def gpt_switch(interaction: discord.Interaction):
    """switching the ChatGPT model between gpt-3.5-turbo and gpt-4.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-switch")
    global model_engine
    if model_engine == "gpt-3.5-turbo":
        model_engine = "gpt-4"
    else:
        model_engine = "gpt-3.5-turbo"
    response = f"モデルエンジンを {model_engine} に変更しました。"
    logger.info("Change the model engine to " + model_engine)
    await interaction.response.send_message(response)


@tree.command(name="gpt-system", description="chat gptのキャラクター設定をする")
async def gpt_system(interaction: discord.Interaction, prompt: str):
    """set up ChatGPT character.

    Args:
        interaction (discord.Interaction): interaction.
        prompt (str): the setting of the ChatGPT character you want it to be.
    """
    logger.info("command: gpt-system")
    global chat_log
    chat_log.append({"role": "system", "content": prompt})
    logger.info("Set gpt character.")
    response = "role: systemを次のように設定しました:\n" + ">>> " + prompt
    await interaction.response.send_message(response)


@client.event
async def on_message(message):
    """
    Process the received message and generate a response.

    Args:
        message: The message object representing the received message.

    Returns:
        None

    Raises:
        Exception: If an error occurs while generating a response.

    """
    global chat_log
    global model_engine
    global total_token
    if message.author.bot:
        return
    if message.author == client.user:
        return
    if str(message.channel.id) == CHANNEL_ID:
        msg = await message.reply("生成中...", mention_author=False)
        # async with message.channel.typing():
        prompt = message.content
        if not prompt:
            await msg.delete()
            await message.channel.send("質問内容がありません")
            return
        chat_log.append({"role": "user", "content": prompt})
        logger.info("user: " + prompt)

        retries = 3
        while retries > 0:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=chat_log,
                )
                response = completion["choices"][0]["message"]["content"]
                response_list = split_string(response)
                chat_log.append(completion["choices"][0]["message"].to_dict())
                logger.info("assistant: " + completion["choices"][0]["message"].to_dict()["content"])
                if model_engine == "gpt-3.5-turbo":
                    price = round_to_digits(
                        completion["usage"]["prompt_tokens"] * 0.0015 / 1000
                        + completion["usage"]["completion_tokens"] * 0.002 / 1000,
                        3,
                    )
                elif model_engine == "gpt-4":
                    price = round_to_digits(
                        completion["usage"]["prompt_tokens"] * 0.03 / 1000
                        + completion["usage"]["completion_tokens"] * 0.06 / 1000,
                        3,
                    )
                    response_list.append(f"(USAGE: {price} USD)")
                logger.info(f"Usage: {price} USD")
                total_token += completion["usage"]["total_tokens"]
                if total_token > 4096:
                    chat_log = chat_log[:1] + chat_log[2:]
                await msg.delete()
                logger.info(response_list)
                for response in response_list:
                    await message.reply(response, mention_author=False)
                break
            except openai.error.InvalidRequestError as e:
                retries -= 1
                if retries > 0:
                    logger.exception(e)
                    await message.reply(
                        f"OpenAI APIでエラーが発生しました。リトライします（残回数{retries}）。\n{traceback.format_exception_only(e)}",
                        mention_author=False,
                    )
                    if "Please reduce the length of the messages." in traceback.format_exception_only(e):
                        chat_log = chat_log[:1] + chat_log[2:]
                else:
                    logger.exception(e)
                    await message.reply(
                        f"OpenAI APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
                    )
            except discord.errors.HTTPException as e:
                logger.exception(e)
                await message.reply(
                    f"Discord APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
                )
                break
            except Exception as e:
                logger.exception(e)
                await message.reply(f"エラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)
                break


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
