from inspect import currentframe
from os import environ
import re

import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from telebot import TeleBot
from telebot.types import Message

from . import bot_reply_markdown

GOOGLE_GEMINI_KEY = environ.get("GOOGLE_GEMINI_KEY")

genai.configure(api_key=GOOGLE_GEMINI_KEY)
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# this prompt copy from https://twitter.com/dotey/status/1737627478007456183
translate_to_chinese_prompt = """
你是一位精通简体中文的专业翻译，尤其擅长将专业学术论文翻译成浅显易懂的科普文章。请你帮我将以下英文段落翻译成中文，风格与中文科普读物相似。

规则：
- 翻译时要准确传达原文的事实和背景。
- 即使上意译也要保留原始段落格式，以及保留术语，例如 FLAC，JPEG 等。保留公司缩写，例如 Microsoft, Amazon, OpenAI 等。
- 人名不翻译
- 同时要保留引用的论文，例如 [20] 这样的引用。
- 对于 Figure 和 Table，翻译的同时保留原有格式，例如：“Figure 1: ”翻译为“图 1: ”，“Table 1: ”翻译为：“表 1: ”。
- 全角括号换成半角括号，并在左括号前面加半角空格，右括号后面加半角空格。
- 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
- 在翻译专业术语时，第一次出现时要在括号里面写上英文原文，例如：“生成式 AI (Generative AI)”，之后就可以只写中文了。
- 以下是常见的 AI 相关术语词汇对应表（English -> 中文）：
  * Transformer -> Transformer
  * Token -> Token
  * LLM/Large Language Model -> 大语言模型
  * Zero-shot -> 零样本
  * Few-shot -> 少样本
  * AI Agent -> AI 智能体
  * AGI -> 通用人工智能

策略：

分三步进行翻译工作，并打印每步的结果：
1. 根据英文内容直译，保持原有格式，不要遗漏任何信息
2. 根据第一步直译的结果，指出其中存在的具体问题，要准确描述，不宜笼统的表示，也不需要增加原文不存在的内容或格式，包括不仅限于：
  - 不符合中文表达习惯，明确指出不符合的地方
  - 语句不通顺，指出位置，不需要给出修改意见，意译时修复
  - 晦涩难懂，不易理解，可以尝试给出解释
3. 根据第一步直译的结果和第二步指出的问题，重新进行意译，保证内容的原意的基础上，使其更易于理解，更符合中文的表达习惯，同时保持原有的格式不变

返回格式如下，"{xxx}"表示占位符：

### 直译
```
{直译结果}
```

***

### 问题
{直译的具体问题列表}

***

### 意译
```
{意译结果}
```

现在请按照上面的要求从第一行开始翻译以下内容为简体中文：
```
"""

# this prompt copy from https://twitter.com/dotey/status/1737732732149457076
translate_to_english_prompt = """
现在我要写一个将中文翻译成英文科研论文的GPT，请参照以下Prompt制作，注意都用英文生成：

## 角色
你是一位科研论文审稿员，擅长写作高质量的英文科研论文。请你帮我准确且学术性地将以下中文翻译成英文，风格与英文科研论文保持一致。

## 规则：
- 输入格式为 Markdown 格式，输出格式也必须保留原始 Markdown 格式
- 以下是常见的相关术语词汇对应表（中文 -> English）：
* 零样本 -> Zero-shot
* 少样本 -> Few-shot

## 策略：

分三步进行翻译工作，并打印每步的结果：
1. 根据中文内容直译成英文，保持原有格式，不要遗漏任何信息
2. 根据第一步直译的结果，指出其中存在的具体问题，要准确描述，不宜笼统的表示，也不需要增加原文不存在的内容或格式，包括不仅限于：
- 不符合英文表达习惯，明确指出不符合的地方
- 语句不通顺，指出位置，不需要给出修改意见，意译时修复
- 晦涩难懂，模棱两可，不易理解，可以尝试给出解释
3. 根据第一步直译的结果和第二步指出的问题，重新进行意译，保证内容的原意的基础上，使其更易于理解，更符合英文科研论文的表达习惯，同时保持原有的格式不变

## 格式
返回格式如下，"{xxx}"表示占位符：

### 直译
```
{直译结果}
```

***

### 问题
{直译的具体问题列表}

***

### 意译
```
{意译结果}
```

现在请按照上面的要求从第一行开始翻译以下内容为英文：
```
"""

# Global history cache
gemini_player_dict = {}


def make_new_gemini_convo():
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    convo = model.start_chat()
    return convo


def gemini_handler(message: Message, bot: TeleBot) -> None:
    """Gemini : /gemini <question>"""
    m = message.text.strip()
    player = None
    player_id = (
        f"{message.chat.id}-{message.from_user.id}-{currentframe().f_code.co_name}"
    )
    # restart will lose all TODO
    if player_id not in gemini_player_dict:
        player = make_new_gemini_convo()
        gemini_player_dict[player_id] = player
    else:
        player = gemini_player_dict[player_id]
    if m.strip() == "clear":
        bot.reply_to(
            message,
            "just clear you gemini messages history",
        )
        player.history.clear()
        return

    # keep the last 5, every has two ask and answer.
    if len(player.history) > 10:
        player.history = player.history[2:]

    try:
        player.send_message(m)
        gemini_reply_text = player.last.text.strip()
        # Gemini is often using ':' in **Title** which not work in Telegram Markdown
        gemini_reply_text = gemini_reply_text.replace(": **", "**\: ")
    except StopCandidateException as e:
        match = re.search(r'content\s*{\s*parts\s*{\s*text:\s*"([^"]+)"', str(e))
        if match:
            gemini_reply_text = match.group(1)
            gemini_reply_text = re.sub(r"\\n", "\n", gemini_reply_text)
        else:
            print("No meaningful text was extracted from the exception.")
            bot.reply_to(
                message,
                "Google gemini encountered an error while generating an answer. Please check the log.",
            )
            return

    # By default markdown
    bot_reply_markdown(message, "Gemini answer", gemini_reply_text, bot)


def gemini_photo_handler(message: Message, bot: TeleBot) -> None:
    s = message.caption
    reply_message = bot.reply_to(
        message,
        "Generating google gemini vision answer please wait.",
    )
    prompt = s.strip()
    # get the high quaility picture.
    max_size_photo = max(message.photo, key=lambda p: p.file_size)
    file_path = bot.get_file(max_size_photo.file_id).file_path
    downloaded_file = bot.download_file(file_path)
    with open("gemini_temp.jpg", "wb") as temp_file:
        temp_file.write(downloaded_file)

    model = genai.GenerativeModel("gemini-pro-vision")
    with open("gemini_temp.jpg", "rb") as image_file:
        image_data = image_file.read()
    contents = {
        "parts": [{"mime_type": "image/jpeg", "data": image_data}, {"text": prompt}]
    }
    try:
        response = model.generate_content(contents=contents)
        bot.reply_to(message, "Gemini vision answer:\n" + response.text)
    finally:
        bot.delete_message(reply_message.chat.id, reply_message.message_id)


def gemini_translate_to_chinese_handler(message: Message, bot: TeleBot) -> None:
    """translate to Chinese with Gemini"""
    m = message.text.strip()
    player = None
    player_id = (
        f"{message.chat.id}-{message.from_user.id}-{currentframe().f_code.co_name}"
    )
    # restart will lose all TODO
    if player_id not in gemini_player_dict:
        player = make_new_gemini_convo()
        gemini_player_dict[player_id] = player
    else:
        player = gemini_player_dict[player_id]
    if m.strip() == "clear":
        bot.reply_to(
            message,
            "just clear you gemini messages history",
        )
        player.history.clear()
        return

    # keep the last 5, every has two ask and answer.
    if len(player.history) > 10:
        player.history = player.history[2:]

    try:
        player.send_message(f"{translate_to_chinese_prompt}\n{m}")
        gemini_reply_text = player.last.text.strip()
        # Gemini is often using ':' in **Title** which not work in Telegram Markdown
        gemini_reply_text = gemini_reply_text.replace(": **", "**\: ")
    except StopCandidateException as e:
        match = re.search(r'content\s*{\s*parts\s*{\s*text:\s*"([^"]+)"', str(e))
        if match:
            gemini_reply_text = match.group(1)
            gemini_reply_text = re.sub(r"\\n", "\n", gemini_reply_text)
        else:
            print("No meaningful text was extracted from the exception.")
            bot.reply_to(
                message,
                "Google gemini encountered an error while generating an answer. Please check the log.",
            )
            return

    # By default markdown
    bot_reply_markdown(message, "Gemini answer", gemini_reply_text, bot)


def gemini_translate_to_english_handler(message: Message, bot: TeleBot) -> None:
    """translate to English with Gemini"""
    m = message.text.strip()
    player = None
    player_id = (
        f"{message.chat.id}-{message.from_user.id}-{currentframe().f_code.co_name}"
    )
    # restart will lose all TODO
    if player_id not in gemini_player_dict:
        player = make_new_gemini_convo()
        gemini_player_dict[player_id] = player
    else:
        player = gemini_player_dict[player_id]
    if m.strip() == "clear":
        bot.reply_to(
            message,
            "just clear you gemini messages history",
        )
        player.history.clear()
        return

    # keep the last 5, every has two ask and answer.
    if len(player.history) > 10:
        player.history = player.history[2:]

    try:
        player.send_message(f"{translate_to_english_prompt}\n{m}")
        gemini_reply_text = player.last.text.strip()
        # Gemini is often using ':' in **Title** which not work in Telegram Markdown
        gemini_reply_text = gemini_reply_text.replace(": **", "**\: ")
    except StopCandidateException as e:
        match = re.search(r'content\s*{\s*parts\s*{\s*text:\s*"([^"]+)"', str(e))
        if match:
            gemini_reply_text = match.group(1)
            gemini_reply_text = re.sub(r"\\n", "\n", gemini_reply_text)
        else:
            print("No meaningful text was extracted from the exception.")
            bot.reply_to(
                message,
                "Google gemini encountered an error while generating an answer. Please check the log.",
            )
            return

    # By default markdown
    bot_reply_markdown(message, "Gemini answer", gemini_reply_text, bot)


def register(bot: TeleBot) -> None:
    bot.register_message_handler(gemini_handler, commands=["gemini"], pass_bot=True)
    bot.register_message_handler(gemini_handler, regexp="^gemini:", pass_bot=True)
    bot.register_message_handler(
        gemini_translate_to_chinese_handler, commands=["t2zh"], pass_bot=True
    )
    bot.register_message_handler(
        gemini_translate_to_chinese_handler, regexp="^t2zh:", pass_bot=True
    )
    bot.register_message_handler(
        gemini_translate_to_english_handler, commands=["t2eng"], pass_bot=True
    )
    bot.register_message_handler(
        gemini_translate_to_english_handler, regexp="^t2eng:", pass_bot=True
    )
    bot.register_message_handler(
        gemini_photo_handler,
        content_types=["photo"],
        func=lambda m: m.caption and m.caption.startswith(("gemini:", "/gemini")),
        pass_bot=True,
    )
