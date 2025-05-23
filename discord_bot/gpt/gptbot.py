import base64
import json
import math
import os
import re
import tempfile
import traceback
from logging import config, getLogger
from typing import Union
from urllib.parse import urlparse

import anthropic
import discord
import openai
import requests
import tomllib
import typing_extensions
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from pdfminer.high_level import extract_text

current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
log_folder_abspath = os.path.join(grandparent_folder_abspath, "logs")
configpath = os.path.join(grandparent_folder_abspath, "pyproject.toml")
paper_folder_abspath = os.path.join(grandparent_folder_abspath, "papers")
paper_chat_logs_json_abspath = os.path.join(paper_folder_abspath, "paper_chat_logs.json")
basename = os.path.basename(__file__).split(".")[0]
with open(configpath, "rb") as f:
    log_conf = tomllib.load(f).get("logging")
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(log_folder_abspath, f"{basename}.log")

logger = getLogger(__name__)
config.dictConfig(log_conf)
dotenvpath = os.path.join(grandparent_folder_abspath, ".env")
load_dotenv(dotenvpath)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN_GPT")
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthoropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_openai_client = openai.OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
) # geminiをopeanaiのクライアントから呼び出す
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CHANNEL_ID = os.getenv("CHANNEL_ID_GPT")
PAPER_CHANNEL_ID = os.getenv("CHANNEL_ID_ARXIVEXPORT")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

model_engine = "gemini-2.0-flash"
chat_log = []
# system = (
    # "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
    # + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
    # + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。"
# )
system = "あなたの名前は「ことのせ つむぐ」で、数理科学を専攻する大学院生である私のAIアシスタントです。"
system_paper = "You are a researcher. You give thoughtful answers, taking into account the back and forth queries."
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


def calculate_price(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message],
    model_engine: str,
) -> float:
    """
    Calculate the price of the OpenAI API or Anthropic API usage based on the completion object.

    The price is calculated based on the number of tokens used in the prompt and
    completion, and the pricing rates for the specified model engine.

    Args:
        completion (openai.types.chat.chat_completion.ChatCompletion): The completion object returned by the OpenAI API.
        model_engine (str): The model engine used for the completion.

    Returns:
        float: The price of the OpenAI API usage in USD.

    Examples:
        >>> calculate_price(completion, "gpt-3.5-turbo")
        0.123
        >>> calculate_price(completion, "gpt-4o")
        0.456
    """
    if model_engine == "gpt-3.5-turbo":
        price = round_to_digits(
            completion.usage.prompt_tokens * 0.50 / 1000000 + completion.usage.completion_tokens * 1.50 / 1000000,
            3,
        )
    elif model_engine == "gpt-4-turbo":
        price = round_to_digits(
            completion.usage.prompt_tokens * 10.00 / 1000000 + completion.usage.completion_tokens * 30.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4o":
        price = round_to_digits(
            completion.usage.prompt_tokens * 2.50 / 1000000 + completion.usage.completion_tokens * 10.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4o-2024-11-20":
        price = round_to_digits(
            completion.usage.prompt_tokens * 2.50 / 1000000 + completion.usage.completion_tokens * 10.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4.1":
        price = round_to_digits(
            completion.usage.prompt_tokens * 2.00 / 1000000 + completion.usage.completion_tokens * 8.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4.1-mini":
        price = round_to_digits(
            completion.usage.prompt_tokens * 0.40 / 1000000 + completion.usage.completion_tokens * 1.60 / 1000000,
            3,
        )
    elif model_engine == "gpt-4.1-nano":
        price = round_to_digits(
            completion.usage.prompt_tokens * 0.10 / 1000000 + completion.usage.completion_tokens * 0.40 / 1000000,
            3,
        )
    elif model_engine == "o4-mini":
        price = round_to_digits(
            completion.usage.prompt_tokens * 1.10 / 1000000 + completion.usage.completion_tokens * 4.40 / 1000000,
            3,
        )
    elif model_engine == "o3":
        price = round_to_digits(
            completion.usage.prompt_tokens * 10.0 / 1000000 + completion.usage.completion_tokens * 40.0 / 1000000,
            3,
        )
    elif model_engine == "gpt-4o-mini":
        price = round_to_digits(
            completion.usage.prompt_tokens * 0.15 / 1000000 + completion.usage.completion_tokens * 0.60 / 1000000,
            3,
        )
    elif model_engine == "claude-3-5-haiku-20241022":
        price = round_to_digits(
            completion.usage.input_tokens * 0.80 / 1000000 + completion.usage.output_tokens * 4.00 / 1000000,
            3,
        )
    elif model_engine == "claude-3-7-sonnet-20250219":
        price = round_to_digits(
            completion.usage.input_tokens * 3.00 / 1000000 + completion.usage.output_tokens * 15.00 / 1000000,
            3,
        )
    elif "gemini" in model_engine:
        price = 0
    return price


def split_string(text: str) -> list[str]:
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
        list[str]: The list of split strings.
    """
    ret_list = []
    buffer = ""
    code_block_flag = False
    for line in text.split("\n"):
        if "```" in line:
            code_block_flag = not code_block_flag
        if len(buffer + line + "\n") <= 2000 or (len(buffer + line + "\n") > 2000 and code_block_flag):
            if code_block_flag and len(buffer + line + "```\n") > 2000:
                ret_list.append(buffer + "```\n")
                buffer = "```\n"
            buffer += line + "\n"
        else:
            ret_list.append(buffer)
            buffer = line + "\n"
    if buffer:
        ret_list.append(buffer)

    ret_list_clean = [s for s in ret_list if s != ""]
    return ret_list_clean


def split_text_by_period(text: str) -> list[str]:
    """
    Split a long string into a list of strings each with maximum length 2000,
    ensuring that splits occur at the end of sentences (periods).

    Args:
        text (str): The string to split.

    Returns:
        list[str]: The list of split strings.
    """
    ret_list = []
    while len(text) > 2000:
        # Find the last period within the first 2000 characters
        split_index = text.rfind(".", 0, 2000)
        if split_index == -1:
            # If no period is found, split at 2000 characters
            split_index = 2000
        else:
            # Include the period in the split
            split_index += 1
        ret_list.append(text[:split_index].strip())
        text = text[split_index:].strip()
    if text:
        ret_list.append(text)
    return ret_list


def get_completion(
    model_engine: str,
    messages: list[str],
    system=None,
    timeout: int = 120,
    max_tokens: int = 16384,
):
    """Get completion from OpenAI API or Anthropic API.

    Args:
        model_engine (str): The model engine to use for the completion.
        messages (list[str]): The list of messages in the conversation.
        system (str): The system message to set for the completion.
        timeout (int): The maximum time in seconds to wait for the completion.
        max_tokens (int): The maximum number of tokens to generate in the completion.

    Returns:
        openai.types.chat.chat_completion.ChatCompletion or anthropic.types.message.Message: The completion object returned by the API.

    Raises:
        None: The function does not raise any exceptions.
    """
    if "gpt" in model_engine:
        system_set = [{"role": "system", "content": system}] if system else []
        return openai_client.chat.completions.create(
            model=model_engine, messages=system_set + messages, timeout=timeout, max_tokens=max_tokens
        )
    elif "o3" in model_engine or "o4-mini" in model_engine:
        return openai_client.chat.completions.create(
            model=model_engine, messages=messages, timeout=timeout
        )
    elif "claude" in model_engine:
        return anthoropic_client.messages.create(
            model=model_engine,
            messages=messages,
            system=system,
            timeout=timeout,
            max_tokens=8192,
        )
    elif "gemini" in model_engine:
        system_set = [{"role": "system", "content": system}] if system else []
        # なんかsystem入れたら毎回名乗り出したので消す
        return gemini_openai_client.chat.completions.create(
            model=model_engine, messages=messages, timeout=timeout
        )

def get_response_text(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message],
) -> str:
    """Get the response text from the completion object.

    Args:
        completion (openai.types.chat.chat_completion.ChatCompletion or anthropic.types.message.Message): The completion object returned by the API.

    Returns:
        str: The response text generated by the model.

    Raises:
        None: The function does not raise any exceptions.
    """
    if isinstance(completion, openai.types.chat.chat_completion.ChatCompletion):
        return completion.choices[0].message.content
    elif isinstance(completion, anthropic.types.message.Message):
        return completion.content[0].text
    elif isinstance(completion, genai.types.GenerateContentResponse):
        return completion.text


def remove_refusal_field(responses):
    # 'refusal'フィールドを削除 これがあるとgeminiのAPIがエラーを返す
    if isinstance(responses, list):
        for response in responses:
            if 'refusal' in response:
                del response['refusal']
    elif isinstance(responses, dict):
        if 'refusal' in responses:
            del responses['refusal']
    return responses

def get_message_log(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message],
):
    if isinstance(completion, openai.types.chat.chat_completion.ChatCompletion):
        return remove_refusal_field(completion.choices[0].message.to_dict())
    elif isinstance(completion, anthropic.types.message.Message):
        return {"role": completion.role, "content": completion.content[0].text}
    elif isinstance(completion, genai.types.GenerateContentResponse):
        return completion.candidates[0]


def get_total_tokens(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message],
) -> int:
    """Get the total number of tokens used in the completion object.

    Args:
        completion (openai.types.chat.chat_completion.ChatCompletion or anthropic.types.message.Message): The completion object returned by the API.

    Returns:
        int: The total number of tokens used in the completion.

    Raises:
        None: The function does not raise any exceptions.
    """
    if isinstance(completion, openai.types.chat.chat_completion.ChatCompletion):
        return completion.usage.total_tokens
    elif isinstance(completion, anthropic.types.message.Message):
        return completion.usage.input_tokens + completion.usage.output_tokens


def format_image_data(base64_image: str, model_engine: str) -> dict:
    """Format the image data for the API response.

    Args:
        base64_image (str): The base64 encoded image data.
        model_engine (str): The model engine to use for the conversion.

    Returns:
        dict: The formatted image data for the API response.

    Raises:
        None: The function does not raise any exceptions.
    """
    is_gpt_or_gemini = ("gpt" in model_engine) or ("o3" in model_engine) or ("o4-mini" in model_engine) or ("gemini" in model_engine)

    if is_gpt_or_gemini:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    else:
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}


def convert_messages(messages: list, model_engine: str):
    """Convert messages to the format required by the API.

    Args:
        messages (list): The list of messages to convert.
        model_engine (str): The model engine to use for the conversion.
    """

    def convert_content(content):
        if isinstance(content, list):
            return [convert_item(item) for item in content]
        return content

    def convert_item(item):
        if item["type"] == "image":
            return format_image_data(item["source"]["data"], model_engine)
        elif item["type"] == "image_url":
            base64_data = item["image_url"]["url"].split(",")[1]
            return format_image_data(base64_data, model_engine)
        return item

    return [{"role": message["role"], "content": convert_content(message["content"])} for message in messages]


async def reply_openai_exception(retries: int, message: Union[discord.Message, discord.Interaction], e: Exception):
    """Handles exceptions that occur during OpenAI API calls and sends appropriate replies.

    Args:
        retries (int): The number of remaining retries.
        message (discord.Message or discord.Interaction): The message or interaction object representing the user's request.
        e (Exception): The exception that occurred during the API call.

    Returns:
        None: The function does not return any value.

    Raises:
        None: The function does not raise any exceptions.
    """
    if retries > 0:
        await message.reply(
            f"OpenAI APIでエラーが発生しました。リトライします（残回数{retries}）。\n{traceback.format_exception_only(e)}",
            mention_author=False,
        )
    else:
        await message.reply(
            f"OpenAI APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
        )


def fix_title_capitalization(title: str):
    # 小文字で残す単語のリスト
    minor_words = {
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "for",
        "nor",
        "on",
        "at",
        "to",
        "by",
        "with",
        "in",
        "of",
        "up",
        "out",
        "as",
    }

    # タイトルをすべて小文字にしたあと単語に分割
    words = title.lower().split()

    # 最初と最後の単語は必ず大文字から始める
    capitalized_title = [
        word.capitalize() if i == 0 or i == len(words) - 1 else (word if word in minor_words else word.capitalize())
        for i, word in enumerate(words)
    ]

    # 結果を結合して新しいタイトルとして返す
    return " ".join(capitalized_title)


def _extract_paper_metadata(clean_text: str):
    content = """
    The opening sentence of the paper is given below. From the given text, correctly extract the title, authors' names and abstract, always following the following notes.
    - In particular, the abstract should be extracted correctly, without word-for-word errors and without any arrangement.
    - Remove any * or number before or after the name of the author(s).
    - To ensure that each author can be distinguished, the authors' names are output separated by ", ".
    - The name of the author(s) may be followed by the name of the organisation to which the author(s) belong, but this must not be extracted.
    - The abstract should be translated into Japanese.
    \n----------------------\n
    """ + clean_text[:16000]

    completion = gemini_client.models.generate_content(
    model="gemini-2.0-flash", contents=content, config={
        'response_mime_type': 'application/json',
        'response_schema': {
            "required": [
                "title",
                "authors",
                "abstract",
                "abstract_ja",
            ],
            "properties": {
                "title": {"type": "STRING"},
                "authors": {"type": "STRING"},
                "abstract": {"type": "STRING"},
                "abstract_ja": {"type": "STRING"},
            },
            "type": "OBJECT",
            }
        },
    )

    response = json.loads(completion.text)
    title = fix_title_capitalization(response["title"])
    authors = response["authors"]
    abstract = response["abstract"]
    abstract_ja = response["abstract_ja"]
    logger.info("assistant: Title: " + title)
    logger.info("assistant: Authors: " + authors)
    logger.info("assistant: Abstract: " + abstract)
    logger.info("assistant: Abstract (Japanese): " + abstract_ja)

    price = calculate_price(completion, "gemini-2.0-flash")
    logger.info(f"Usage: {price} USD, responsed by gemini-2.0-flash)")

    return {"title": title, "authors": authors, "abstract": abstract, "abstract_ja": abstract_ja, "price": price}


async def _send_metadata_to_thread(thread, metadata):
    response_list = [
        f"**Authors**: {metadata['authors']}",
        f"**Abstract**: {metadata['abstract']}",
        f"**あらまし**: {metadata['abstract_ja']}",
        f"(USAGE: {metadata['price']} USD, responsed by gemini-2.0-flash)",
    ]
    for response in response_list:
        # もしresponseが長い場合は分割して送信する
        if len(response) > 2000:
            split_responses = split_text_by_period(response)
            for split_response in split_responses:
                await thread.send(split_response)
        else:
            await thread.send(response)


async def _summary_paper_content(thread, clean_text):
    content = f"""日本語で答えてください。
    ユーザーから与えられた論文の内容について、60秒で読めるように、以下のすべての問いに一問一答で答えてください。ただし、専門用語だと思われるものを日本語に翻訳した場合は、翻訳前の用語もあわせて記してください。各回答内では箇条書きの使用は禁止です。各回答内では改行してはいけません。各回答の間には改行を含めてください。
    \n----------------------\n
    {clean_text}
    \n----------------------\n
    【目的】この論文の目的は何？
    【問題意識】先行研究ではどのような点が課題だったか？
    【手法】この研究の手法は何？独自性は？
    【結果】どのように有効性を定量、定性的に評価したか？
    【限界】この論文における限界は？
    【次の研究の方向性】次に読むべき論文は？（論文番号があれば、それもあわせて）
    """

    global system_paper
    messages = [{"role": "user", "content": content}]
    message = await thread.send("生成中...")

    retries = 3
    while retries > 0:
        try:
            completion = gemini_client.models.generate_content(model="gemini-2.0-flash",contents=content)
            summary_response = get_response_text(completion)
            logger.info("assistant: " + summary_response)
            response_list = split_string(summary_response)
            price = calculate_price(completion, "gemini-2.0-flash")
            logger.info(f"Usage: {price} USD, responsed by gemini-2.0-flash")
            response_list.append(f"(USAGE: {price} USD, responsed by gemini-2.0-flash)")
            messages.append({"role": "assistant", "content": summary_response})
            await message.delete()
            for response in response_list:
                await thread.send(response)
            global paper_chat_logs
            paper_chat_logs[str(thread.id)] = messages
            with open(paper_chat_logs_json_abspath, "w") as f:
                json.dump(paper_chat_logs, f, ensure_ascii=False)
            return summary_response
        except Exception as e:
            retries -= 1
            logger.exception(e)
            await thread.send(f"Gemini APIでエラーが発生しました。リトライします（残回数{retries}）。\n{traceback.format_exception_only(e)}")


async def _extract_challenging_words_for_english_learners(thread, clean_text: str):
    content = """
    Identify at least 10 words in the text of this paper that might be challenging for English learners. For each word:
    - Provide the English word.
    - Give its Japanese translation.
    - Create or extract an example sentence using only one word from the list.
    - Include the Japanese translation of the example sentence.

    Make sure that the target English word and its Japanese translation are **bold** in the example sentences. Example sentences can come from the paper itself or be original.
    \n----------------------\n
    """ + clean_text

    completion = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=content,
        config={
            'response_mime_type': 'application/json',
            'response_schema': {
                'type': 'ARRAY',
                "items": {
                    'required': [
                        'en_word',
                        'ja_translation',
                        'example_sentence',
                        'ja_example_sentence',
                    ],
                    'properties': {
                        'en_word': {'type': 'STRING'},
                        'ja_translation': {'type': 'STRING'},
                        'example_sentence': {'type': 'STRING'},
                        'ja_example_sentence': {'type': 'STRING'},
                    },
                    'type': 'OBJECT',
                }
            },
        },
    )
    response = json.loads(completion.text)
    blocks = []
    for res in response:
        en_word = res['en_word']
        ja_translation = res['ja_translation']
        example_sentence = res['example_sentence']
        ja_example_sentence = res['ja_example_sentence']
        block = (
            f"- {en_word}: {ja_translation}\n"
            f"  - **例文**: {example_sentence}\n"
            f"  - **訳**: {ja_example_sentence}\n"
        )
        blocks.append(block)

    # 2000文字以下になるようにブロックを結合
    combined_responses = []
    current_response = ""
    for block in blocks:
        if len(current_response) + len(block) > 2000:
            combined_responses.append(current_response)
            current_response = block
        else:
            current_response += block
    if current_response:
        combined_responses.append(current_response)

    # 分割されたレスポンスを送信
    try:
        for response in combined_responses:
            await thread.send(response)
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        await thread.send(error_message)

async def _extract_keywords_from_paper(thread, clean_text: str):
    content = """
    Identify at least 4 keywords from the text of this paper in lowercase.

    Make sure that the keywords are relevant to the main topics of the paper.
    \n----------------------\n
    """ + clean_text

    completion = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=content,
        config={
            'response_mime_type': 'application/json',
            'response_schema': {
                'type': 'ARRAY',
                "items": {
                    'required': [
                        'keyword',
                    ],
                    'properties': {
                        'keyword': {'type': 'STRING'},
                    },
                    'type': 'OBJECT',
                }
            },
        },
    )
    response = json.loads(completion.text)
    keywords_list = [{"name": res['keyword']} for res in response]
    keywords = "**Keywords**: " + ", ".join([res['name'] for res in keywords_list])

    # 分割されたレスポンスを送信
    try:
        await thread.send(keywords)
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        await thread.send(error_message)
    
    return keywords_list

@client.event
async def on_ready():
    """on ready"""
    print(f"We have logged in as {client.user}")
    await tree.sync()


@tree.command(name="gpt-hflush", description="chat gptのチャット履歴を消去する")
async def gpt_delete(interaction: discord.Interaction):
    """delete chat history with ChatGPT.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-hflush")
    global chat_log
    global total_token
    chat_log = []
    total_token = 0
    logger.info("Deleted chat logs.")
    response = "チャット履歴を削除しました。"
    await interaction.response.send_message(response)


@tree.command(name="gpt-switch", description="使用するチャットモデルを設定します")
@app_commands.describe(model="使用するモデルを選択してください")
@app_commands.choices(model=[
    app_commands.Choice(name="GPT-4o", value="gpt-4o"),
    app_commands.Choice(name="GPT-4o-2024-11-20", value="gpt-4o-2024-11-20"),
    app_commands.Choice(name="GPT-4o-mini", value="gpt-4o-mini"),
    app_commands.Choice(name="GPT-4.1", value="gpt-4.1"),
    app_commands.Choice(name="GPT-4.1-mini", value="gpt-4.1-mini"),
    app_commands.Choice(name="GPT-4.1-nano", value="gpt-4.1-nano"),
    app_commands.Choice(name="o4-mini", value="o4-mini"),
    app_commands.Choice(name="o3", value="o3"),
    app_commands.Choice(name="Gemini-2.0-flash", value="gemini-2.0-flash"),
    app_commands.Choice(name="Gemini-2.0-flash-thinking-exp-01-21", value="gemini-2.0-flash-thinking-exp-01-21"),
    app_commands.Choice(name="Gemini-2.5-pro-exp-03-25", value="gemini-2.5-pro-exp-03-25"),
    app_commands.Choice(name="Gemini-2.5-flash-preview-04-17", value="gemini-2.5-flash-preview-04-17"),
    app_commands.Choice(name="Claude-3.5-haiku-20241022", value="claude-3-5-haiku-20241022"),
    app_commands.Choice(name="Claude-3.7-sonnet-20250219", value="claude-3-7-sonnet-20250219"),
])
async def set_model(interaction: discord.Interaction, model: app_commands.Choice[str]):
    global model_engine, chat_log
    model_engine = model.value
    chat_log = convert_messages(chat_log, model_engine)
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
    global system
    system = prompt
    logger.info("Set gpt character.")
    response = "role: systemを次のように設定しました:\n" + ">>> " + prompt
    await interaction.response.send_message(response)


def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text


def clean_extracted_text(text):
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text


async def is_valid_link(url: str) -> bool:
    """
    指定されたURLが有効かどうかを確認する。

    Args:
        url (str): チェックするURL。

    Returns:
        bool: 有効な場合はTrue、無効な場合はFalse。
    """
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"リンクのチェック中にエラーが発生しました: {e}")
        return False


class PaperMetadata(typing_extensions.TypedDict):
    title: str
    authors: str
    abstract: str
    abstract_ja: str


paper_chat_logs = {}
# load paper_chat_logs
if os.path.exists(paper_chat_logs_json_abspath):
    with open(paper_chat_logs_json_abspath, "r") as f:
        paper_chat_logs = json.load(f)


@tree.command(name="gpt-paper-interpreter", description="論文の要約を行います。建てられたスレッド内で対話も可能です。")
async def paper_interpreter(interaction: discord.Interaction, pdf_file: discord.Attachment):
    """Summarizes the paper and starts a thread for further discussion.

    Args:
        interaction (discord.Interaction): The interaction object representing the user's request.
        pdf_file (discord.Attachment): The PDF file to be summarized.

    Returns:
        None: The function does not return any value.
    """
    await interaction.response.defer()

    logger.info("command: gpt-interpreter")
    # mydir = "/home/raspberrypi4/discord_bot/pdfs"
    file_path = f"{paper_folder_abspath}/{pdf_file.filename}"
    await pdf_file.save(file_path)

    logger.info("saved " + file_path)

    extracted_text = extract_text_from_pdf(file_path)
    clean_text = clean_extracted_text(extracted_text)

    metadata = _extract_paper_metadata(clean_text)
    title = metadata["title"]

    message = await interaction.followup.send(f"**Title**: {title}")
    thread = await interaction.channel.create_thread(
        name=title if len(title) < 100 else title[:97] + "...", message=message, auto_archive_duration=60
    )

    await _send_metadata_to_thread(thread, metadata)
    keywords_list = await _extract_keywords_from_paper(thread, clean_text)
    summary = await _summary_paper_content(thread, clean_text)
    await _extract_challenging_words_for_english_learners(thread, clean_text)
    if NOTION_TOKEN is not None:
        thread_url = thread.jump_url
        post_to_notion_database(metadata, "", thread_url, summary, keywords_list)


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
    if message.author.bot:
        return
    if message.author == client.user:
        return

    # Handle messages in PAPER_CHANNEL_ID
    if str(message.channel.id) == PAPER_CHANNEL_ID:
        # Check for PDF attachments first
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(".pdf") or \
                   attachment.content_type == "application/pdf":
                    await on_paper_attachment(message, attachment)
                    return  # Process only the first PDF attachment

        # If no PDF attachment processed, check for links
        if any(
            link_keyword in message.content
            for link_keyword in [
                "https://arxiv.org",
                "https://proceedings.mlr.press",
                "https://openreview.net",
                "https://proceedings.neurips.cc",
                ".pdf",
            ]
        ):
            # Further validation (e.g. message.content.startswith("http")) is in on_paper_link
            await on_paper_link(message)
            return # Link processed

    # Handle messages in paper discussion threads
    if (
        message.channel.type == discord.ChannelType.public_thread
        and message.channel.owner_id == client.user.id
        and str(message.channel.id) in paper_chat_logs.keys()
    ):
        await on_paper_thread(message)
        return # Thread message processed

    # Handle messages in the general GPT channel
    if str(message.channel.id) == CHANNEL_ID:
        await on_gpt_channel_response(message)
        # No return needed if this is the last check


async def on_paper_link(message):
    # メッセージがリンクのみであることを確認
    msg = await message.reply("生成中...", mention_author=False)
    if not message.content.startswith("http"): # Ensure it's a link
        await msg.edit(content="メッセージに有効なリンクが含まれていません。処理を中断します。")
        return

    raw_link = message.content
    pdf_link = raw_link
    if "https://arxiv.org" in raw_link:
        pdf_link = message.content.replace("abs", "pdf")
    elif "https://proceedings.mlr.press" in raw_link:
        if "html" in raw_link:
            parts = raw_link.rsplit("/", 1)
            part_head = parts[0]
            part_tail = parts[-1].replace(".html", "")
            pdf_link = f"{part_head}/{part_tail}/{part_tail}.pdf"
    elif "https://openreview.net" in raw_link:
        if "pdf" not in raw_link:
            pdf_link = message.content.replace("forum", "pdf")
    elif "https://proceedings.neurips.cc" in raw_link:
        if "html" in raw_link:
            pdf_link = raw_link.replace("/hash/", "/file/")
            pdf_link = pdf_link.replace("Abstract-Conference.html", "Paper-Conference.pdf")

    file_path = ""  # Initialize file_path
    # PDFリンクの有効性をチェック
    if pdf_link and await is_valid_link(pdf_link):
        logger.info(f"有効なPDFリンクを取得: {pdf_link}")
        pdf_response = requests.get(pdf_link)
        
        # Extract filename more robustly
        parsed_url = urlparse(pdf_link)
        file_name = os.path.basename(parsed_url.path)
        if not file_name or not file_name.lower().endswith((".pdf", ".html")): # Check if filename is reasonable
             # Try to get from content-disposition or generate one
            cd = pdf_response.headers.get('content-disposition')
            if cd:
                fname = re.findall('filename="?([^"]+)"?', cd)
                if fname:
                    file_name = fname[0]
            if not file_name or not file_name.lower().endswith(".pdf"): # If still no good name
                # Fallback for names not ending with .pdf from URL path
                potential_name_from_url = pdf_link.split("/")[-1]
                if potential_name_from_url.lower().endswith(".pdf"):
                    file_name = potential_name_from_url
                else: # Final fallback
                    file_name = "downloaded_paper.pdf"


        if not file_name.lower().endswith(".pdf"): # Ensure .pdf extension if it's missing
             file_name_base, file_ext = os.path.splitext(file_name)
             if file_ext.lower() == ".html" and ".pdf" not in file_name_base.lower(): # if it was an html link leading to pdf
                 file_name = file_name_base + ".pdf"
             elif not file_ext: # no extension
                 file_name = file_name + ".pdf"


        file_path = os.path.join(paper_folder_abspath, file_name)
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(pdf_response.content)
        logger.info("download and saved " + file_path)
    else:
        logger.warning(f"無効なPDFリンク: {pdf_link}")
        await msg.edit(content="PDFリンクの取得または検証に失敗しました。処理を中断します。")
        return

    extracted_text = extract_text_from_pdf(file_path)
    clean_text = clean_extracted_text(extracted_text)
    
    await msg.delete() # Delete the "生成中..." message
    await _process_paper_and_reply(message, clean_text, raw_link)


async def on_paper_attachment(message: discord.Message, attachment: discord.Attachment):
    """Handles PDF file attachments in the paper channel."""
    msg = await message.reply("添付ファイルを処理中...", mention_author=False)
    try:
        file_path = os.path.join(paper_folder_abspath, attachment.filename)
        await attachment.save(file_path)
        logger.info(f"Saved attached PDF: {file_path}")

        extracted_text = extract_text_from_pdf(file_path)
        clean_text = clean_extracted_text(extracted_text)

        await msg.delete()  # Delete the "処理中..." message
        # For attachments, raw_link is not applicable, so we pass an empty string or omit it if using default
        await _process_paper_and_reply(message, clean_text, raw_link="")
    except Exception as e:
        logger.exception(f"Error processing attached PDF: {e}")
        await msg.edit(content=f"添付されたPDFの処理中にエラーが発生しました。\n{traceback.format_exception_only(e)}")


def split_text(text, max_length=2000):
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def format_chunk(chunk):
    # **text** の形式を検出して太字にする
    bold_pattern = re.compile(r"\*\*(.*?)\*\*")
    parts = []
    last_end = 0

    for match in bold_pattern.finditer(chunk):
        start, end = match.span()
        # 通常のテキスト部分を追加
        if start > last_end:
            parts.append({"type": "text", "text": {"content": chunk[last_end:start]}})
        # 太字部分を追加
        parts.append({"type": "text", "text": {"content": match.group(1)}, "annotations": {"bold": True}})
        last_end = end

    # 残りのテキストを追加
    if last_end < len(chunk):
        parts.append({"type": "text", "text": {"content": chunk[last_end:]}})

    return parts


def post_to_notion_database(metadata, link: str, thread_url: str, summary: str, keywords_list: list[str]):
    """
    Posts the given information to a Notion database.

    Args:
        link (str): Holds the URL link to the original content.
        thread_url (str): Provides the URL link to the Discord thread.
        summary (str): Includes the summary of the paper content.

    Returns:
        None

    Raises:
        None
    """
    title = metadata["title"]
    authors = metadata["authors"]
    abstract_ja = metadata["abstract_ja"]

    notion_page_url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    properties = {
        "名前": {"title": [{"text": {"content": f"{title}"}}]},
        "Author": {"rich_text": [{"text": {"content": f"{authors}"}}]},
        "タグ": {
            "multi_select": keywords_list  # Assuming keywords_list is already in the correct format [{"name": "keyword1"}, ...]
        }
    }

    # Dynamically build the rich_text array for the 'link' property
    link_rich_text_elements = []
    if link:  # Only add if link (raw_link) is not empty
        link_rich_text_elements.append({"text": {"content": link, "link": {"url": link}}})

    if thread_url:
        if link_rich_text_elements:  # If link element already exists, add a separator
            link_rich_text_elements.append({"text": {"content": ", "}})
        link_rich_text_elements.append({"text": {"content": thread_url, "link": {"url": thread_url}}})

    # Only add the 'link' property if there are elements to include
    if link_rich_text_elements:
        properties["link"] = {"rich_text": link_rich_text_elements}
    
    # Prepare the complete data payload for Notion API
    data_payload = {
        "parent": {"database_id": f"{NOTION_DATABASE_ID}"},
        "properties": properties
    }
    
    data = json.dumps(data_payload)

    logger.info(f"Notion API Request Data: {data}") # Log the request data for debugging
    response = requests.post(notion_page_url, headers=headers, data=data)
    logger.info(f"Notion API Response Status: {response.status_code}") # Log status
    logger.info(f"Notion API Response Text: {response.text}") # Log full response text for debugging

    if response.status_code == 200:
        notion_page_id = response.json().get("id")
        notion_page_url = f"https://www.notion.so/{notion_page_id.replace('-', '')}"
        logger.info(f"Notionページが作成されました: {notion_page_url}")

        # abstract_jaとsummaryを2000文字以下に分割
        abstract_ja = "**あらまし**: " + abstract_ja
        abstract_chunks = split_text(abstract_ja)
        summary_lines = summary.split("\n")
        summary_chunks = [chunk for line in summary_lines for chunk in split_text(line)]

        # summaryを追加するためのデータ
        summary_data = json.dumps(
            {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "Abstract"}}]},
                    },
                    *[
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": format_chunk(chunk)
                            },
                        }
                        for chunk in abstract_chunks
                    ],
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"text": {"content": "Summary"}}]},
                    },
                    *[
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": format_chunk(chunk)
                            },
                        }
                        for chunk in summary_chunks
                    ],
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": "(summarized by gemini-2.0-flash)"}}]},
                    },
                ]
            }
        )

        # Notion APIを使ってsummaryを追加
        summary_url = f"https://api.notion.com/v1/blocks/{notion_page_id}/children"
        summary_response = requests.patch(summary_url, headers=headers, data=summary_data)

        if summary_response.status_code == 200:
            logger.info("Notionページにsummaryが追加されました。")
        else:
            logger.error(f"Notionページへのsummary追加に失敗しました: {summary_response.text}")
    else:
        logger.info(f"Notionページへのエクスポートに失敗しました:\n {response}")


async def extract_message_content(message):
    prompt = message.content
    attachments = message.attachments
    content = []
    if prompt:
        content.append({"type": "text", "text": f"{prompt}"})
    if len(attachments) > 0:
        for attachment in attachments:
            if attachment.content_type.startswith("image"):
                # 画像のダウンロード
                image_data = await attachment.read()
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(image_data)
                img_path = temp_file.name
                # base64
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    content.append(format_image_data(base64_image, model_engine))
            if "text/plain" in attachment.content_type:
                # テキストのダウンロード
                text_data = await attachment.read()
                text = text_data.decode("utf-8")
                logger.info(text)
                content.append({"type": "text", "text": f"{text}"})
            if attachment.content_type == "application/pdf":
                # PDFのダウンロード
                pdf_data = await attachment.read()
                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(pdf_data)
                pdf_path = temp_file.name
                # テキスト抽出
                text = clean_extracted_text(extract_text_from_pdf(pdf_path))
                content.append({"type": "text", "text": f"{text}"})
    return content


async def on_gpt_channel_response(message):
    global chat_log
    global paper_chat_logs
    global model_engine
    global total_token

    msg = await message.reply("生成中...", mention_author=False)
    prompt = message.content
    if not prompt and not message.attachments:
        await msg.delete()
        await message.channel.send("質問内容がありません")
        return
    content = await extract_message_content(message)
    chat_log.append({"role": "user", "content": content})
    logger.info(f"user: {content}")
    retries = 3
    retries = 3
    while retries > 0:
        try:
            completion = get_completion(model_engine, chat_log, system=system, timeout=240)
            response = get_response_text(completion)
            response_list = split_string(response)
            chat_log.append(get_message_log(completion))
            logger.info("assistant: " + response)
            price = calculate_price(completion, model_engine)
            response_list.append(f"(USAGE: {price} USD, responsed by {model_engine})")
            logger.info(f"Usage: {price} USD, responsed by {model_engine}")
            total_token += get_total_tokens(completion)
            if "gpt-4o" in model_engine and total_token > 128000 - 256:
                chat_log = chat_log[1:]
            elif "4.1" in model_engine and total_token > 1047576 - 256:
                chat_log = chat_log[1:]
            elif "gemini" in model_engine and total_token > 1000000 - 256:
                chat_log = chat_log[1:]
            elif (("claude" in model_engine) or model_engine == "o3" or model_engine == "o4-mini") and total_token > 200000 - 256:
                chat_log = chat_log[1:]
            logger.info(chat_log)
            # logger.debug(completion)
            await msg.delete()
            for response in response_list:
                await message.reply(response, mention_author=False)
            break
        except openai.APITimeoutError as e:
            retries -= 1
            logger.exception(e)
            await reply_openai_exception(retries, message, e)
        except openai.BadRequestError as e:
            retries -= 1
            logger.exception(e)
            await reply_openai_exception(retries, message, e)
            chat_log = chat_log[1:]
        except discord.errors.HTTPException as e:
            logger.exception(e)
            await message.reply(
                f"Discord APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
            )
            break
        except Exception as e:
            retries -= 1 
            logger.exception(e)
            await message.reply(f"エラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)
            if retries == 0:
                break

async def on_paper_thread(message):
    """Process the received message and generate a response.

    Args:
        message (discord.Message): The message object representing the received message.

    Returns:
        None
    """
    global paper_chat_logs
    global model_engine
    global system_paper

    msg = await message.reply("生成中...", mention_author=False)
    content = await extract_message_content(message)
    paper_chat_logs[str(message.channel.id)].append({"role": "user", "content": content})
    retries = 3
    while retries > 0:
            try:
                try:
                    logger.info({"role": "user", "content": content})
                    completion = get_completion(
                        model_engine,
                        remove_refusal_field(paper_chat_logs[str(message.channel.id)]),
                        system=system_paper,
                        timeout=240
                    )
                except openai.APITimeoutError as e:
                    logger.info("Timeout error during get_completion")
                    retries -= 1
                    logger.exception(e)
                    await reply_openai_exception(retries, message, e)
                    continue
                except Exception as e:
                    logger.exception("Error during get_completion")
                    await message.reply(f"get_completionでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)
                    break

                response = get_response_text(completion)
                response_list = split_string(response)
                paper_chat_logs[str(message.channel.id)].append(get_message_log(completion))
                with open(paper_chat_logs_json_abspath, "w") as f:
                    json.dump(paper_chat_logs, f, ensure_ascii=False)
                logger.info("assistant: " + response)
                price = calculate_price(completion, model_engine)
                logger.info(f"Usage: {price} USD, responsed by {model_engine}")
                response_list.append(f"(USAGE: {price} USD, responsed by {model_engine})")
                await msg.delete()
                for response in response_list:
                    await message.channel.send(response)
                break
            except Exception as e:
                logger.exception(e)
                await message.reply(f"エラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)
                break


async def _process_paper_and_reply(message: discord.Message, clean_text: str, raw_link: str = ""):
    """
    Processes the cleaned text of a paper, creates a thread, sends metadata, summary, etc.

    Args:
        message (discord.Message): The message object representing the received message.
        clean_text (str): The cleaned text of the paper.
        raw_link (str): The raw link to the paper.

    Returns:
        None
    """
    metadata = _extract_paper_metadata(clean_text)
    title = metadata["title"]

    message = await message.reply(f"**Title**: {title}")
    thread = await message.channel.create_thread(
        name=title if len(title) < 100 else title[:97] + "...", message=message, auto_archive_duration=60
    )

    await _send_metadata_to_thread(thread, metadata)
    keywords_list = await _extract_keywords_from_paper(thread, clean_text)
    summary = await _summary_paper_content(thread, clean_text)
    await _extract_challenging_words_for_english_learners(thread, clean_text)
    if NOTION_TOKEN is not None:
        thread_url = thread.jump_url
        post_to_notion_database(metadata, raw_link, thread_url, summary, keywords_list)


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
