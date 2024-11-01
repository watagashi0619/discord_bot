import base64
import json
import math
import os
import re
import tempfile
import tomllib
import traceback
from logging import config, getLogger
from typing import Union

import anthropic
import discord
import openai
from discord import app_commands
from dotenv import load_dotenv
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
CHANNEL_ID = os.getenv("CHANNEL_ID_GPT")
model_engine = "gpt-4o-2024-08-06"
# model_engines = ["gpt-4o-mini","gpt-4o"]
chat_log = []
# system_set = [
#     {
#         "role": "system",
#         "content": "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
#         + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
#         + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。",
#     }
# ]
system = (
    "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
    + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
    + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。"
)
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
            completion.usage.prompt_tokens * 5.00 / 1000000 + completion.usage.completion_tokens * 15.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4o-2024-08-06":
        price = round_to_digits(
            completion.usage.prompt_tokens * 2.50 / 1000000 + completion.usage.completion_tokens * 10.00 / 1000000,
            3,
        )
    elif model_engine == "gpt-4o-mini":
        price = round_to_digits(
            completion.usage.prompt_tokens * 0.15 / 1000000 + completion.usage.completion_tokens * 0.60 / 1000000,
            3,
        )
    elif model_engine == "claude-3-5-sonnet-20240620":
        price = round_to_digits(
            completion.usage.input_tokens * 3.00 / 1000000 + completion.usage.output_tokens * 15.00 / 1000000,
            3,
        )
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


def get_completion(
    model_engine: str,
    messages: list[str],
    system=None,
    timeout: int = 120,
    max_tokens: int = 4096,
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
    elif "claude" in model_engine:
        return anthoropic_client.messages.create(
            model=model_engine,
            messages=messages,
            system=system,
            timeout=timeout,
            max_tokens=max_tokens,
        )


def get_response_text(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message]
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


def get_message_log(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message]
):
    if isinstance(completion, openai.types.chat.chat_completion.ChatCompletion):
        return completion.choices[0].message.to_dict()
    elif isinstance(completion, anthropic.types.message.Message):
        return {"role": completion.role, "content": completion.content[0].text}


def get_total_tokens(
    completion: Union[openai.types.chat.chat_completion.ChatCompletion, anthropic.types.message.Message]
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
    is_gpt = "gpt" in model_engine

    if is_gpt:
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


@tree.command(name="gpt-switch", description="chat gptモデルをgpt-4o-miniとgpt-4oの間で切り替える")
async def gpt_switch(interaction: discord.Interaction):
    """switching the ChatGPT model between gpt-4o-mini and gpt-4o.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-switch")
    global model_engine
    global chat_log
    if model_engine == "gpt-4o-mini":
        model_engine = "gpt-4o-2024-08-06"
    elif model_engine == "gpt-4o-2024-08-06":
        model_engine = "gpt-4o"
    elif model_engine == "gpt-4o":
        model_engine = "claude-3-5-sonnet-20240620"
    elif model_engine == "claude-3-5-sonnet-20240620":
        model_engine = "gpt-4o-mini"
    response = f"モデルエンジンを {model_engine} に変更しました。"
    logger.info("Change the model engine to " + model_engine)
    chat_log = convert_messages(chat_log, model_engine)
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

    # extractor = PaperMetaInfo()
    # title, abstract = extractor.get_title_and_abstract(file_path)

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "The opening sentence of the paper is given as following text. From the given text, extract the title, authors' names and abstract correctly. In particular, extract the abstract correctly without making a single mistake and without making any arrangements. Also, translate the extracted abstract into Japanese.\n\n###\n\n"
                + clean_text[:16000],
            }
        ],
        functions=[
            {
                "name": "get_metadata",
                "description": "extract the title, authors' names and abstract correctly. Also, translate the extracted abstract into Japanese.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "title"},
                        "authors": {"type": "string", "description": "authors"},
                        "abstract": {"type": "string", "description": "abstract"},
                        "abstract_ja": {"type": "string", "description": "abstract translated into Japanese"},
                    },
                    "required": ["title", "authors", "abstract", "abstract_ja"],
                },
            }
        ],
        function_call="auto",
    )
    response = json.loads(completion.choices[0].message.function_call.arguments)
    title = response["title"]
    authors = response["authors"]
    abstract = response["abstract"]
    abstract_ja = response["abstract_ja"]
    logger.info("assistant: Title: " + title)
    logger.info("assistant: Authors: " + authors)
    logger.info("assistant: Abstract: " + abstract)
    logger.info("assistant: Abstract (Japanese): " + abstract_ja)

    price = calculate_price(completion, "gpt-4o-mini")
    logger.info(f"Usage: {price} USD")

    message = await interaction.followup.send(f"**Title**: {title}")
    thread = await interaction.channel.create_thread(
        name=title if len(title) < 100 else title[:97] + "...", message=message, auto_archive_duration=60
    )

    response_list = [
        f"**Authors**: {authors}",
        f"**Abstract**: {abstract}",
        f"**あらまし**: {abstract_ja}",
        f"(USAGE: {price} USD)",
    ]
    for response in response_list:
        await thread.send(response)

    content = f"""日本語で答えてください。
    ユーザーから与えられた論文の内容について、60秒で読めるように、以下のすべての問いに一問一答で答えてください。ただし、専門用語だと思われるものを日本語に翻訳した場合は、翻訳前の原文もあわせて記してください。
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
            model_for_paper = "gpt-4o-2024-08-06"
            completion = get_completion(model_for_paper, messages, system=system_paper, timeout=300, max_tokens=4096)
            response = get_response_text(completion)
            logger.info("assistant: " + response)
            response_list = split_string(response)
            price = calculate_price(completion, model_for_paper)
            logger.info(f"Usage: {price} USD")
            response_list.append(f"(USAGE: {price} USD)")
            messages.append(get_message_log(completion))
            await message.delete()
            for response in response_list:
                await thread.send(response)
            break
        except openai.APITimeoutError as e:
            retries -= 1
            logger.exception(e)
            await reply_openai_exception(retries, thread, e)

    global paper_chat_logs
    paper_chat_logs[str(thread.id)] = messages
    # export paper_chat_logs
    with open(paper_chat_logs_json_abspath, "w") as f:
        json.dump(paper_chat_logs, f, ensure_ascii=False)


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
    global paper_chat_logs
    global model_engine
    global total_token
    if message.author.bot:
        return
    if message.author == client.user:
        return
    if (
        message.channel.type == discord.ChannelType.public_thread
        and message.channel.owner_id == client.user.id
        and str(message.channel.id) in paper_chat_logs.keys()
    ):
        await on_paper_thread(message)
    if str(message.channel.id) == CHANNEL_ID:
        msg = await message.reply("生成中...", mention_author=False)
        # async with message.channel.typing():
        prompt = message.content
        if not prompt and not message.attachments:
            await msg.delete()
            await message.channel.send("質問内容がありません")
            return
        content = []
        if prompt:
            content.append({"type": "text", "text": f"{prompt}"})
        if len(message.attachments) > 0:
            for attachment in message.attachments:
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
                    text = extract_text_from_pdf(pdf_path)
                    content.append({"type": "text", "text": f"{text}"})
        chat_log.append({"role": "user", "content": content})
        logger.info(f"user: {content}")
        retries = 3
        while retries > 0:
            try:
                completion = get_completion(model_engine, chat_log, system=system, timeout=120, max_tokens=4096)
                response = get_response_text(completion)
                response_list = split_string(response)
                chat_log.append(get_message_log(completion))
                logger.info("assistant: " + response)
                price = calculate_price(completion, model_engine)
                response_list.append(f"(USAGE: {price} USD, responsed by {model_engine})")
                logger.info(f"Usage: {price} USD, responsed by {model_engine}")
                total_token += get_total_tokens(completion)
                if model_engine == "gpt-4o" and total_token > 128000 - 256:
                    chat_log = chat_log[1:]
                elif model_engine == "gpt-4o-mini" and total_token > 128000 - 256:
                    chat_log = chat_log[1:]
                elif model_engine == "claude-3-5-sonnet-20240620" and total_token > 200000 - 256:
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
                logger.exception(e)
                await message.reply(
                    f"エラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
                )
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
    prompt = message.content
    paper_chat_logs[str(message.channel.id)].append({"role": "user", "content": prompt})
    retries = 3
    while retries > 0:
        try:
            completion = get_completion(
                model_engine,
                paper_chat_logs[str(message.channel.id)],
                system=system_paper,
                timeout=240,
                max_tokens=4096,
            )
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
        except openai.APITimeoutError as e:
            logger.info(e)
            retries -= 1
            logger.exception(e)
            await reply_openai_exception(retries, message, e)


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
