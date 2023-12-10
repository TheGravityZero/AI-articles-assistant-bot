from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import typing as t

from model.model import AI_Assistant
from settings import BOT_TOKEN, INFERENCE_SETTINGS, logger

bot = Bot(token=BOT_TOKEN)
ai_assistant = AI_Assistant()
dp = Dispatcher(settings=INFERENCE_SETTINGS, assistant=ai_assistant)


@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Ask any question")


@dp.message(Command("help"))
async def get_help(message: types.Message):
    await message.answer(
      "Type any question. You can use /enable_rag and /disable_rag to turn on/off rag"
    )


@dp.message(Command("enable_rag"))
async def enable_database(message: types.Message, settings: t.Dict[t.Any, t.Any]):
    settings['use_rag'] = True
    await message.answer("RAG enabled")


@dp.message(Command("disable_rag"))
async def disable_database(message: types.Message, settings: t.Dict[t.Any, t.Any]):
    settings['use_rag'] = False
    await message.answer("RAG disabled")


@dp.message()
async def get_answer(message: types.Message, settings: t.Dict[t.Any, t.Any], assistant: AI_Assistant):
    try:
        query = message.text.strip()
        logger.info(f"Got request: {query}")
        logger.info(f"Generating answer. RAG flag: {settings.get('use_rag')}")
        answer = assistant.generate_response(query, settings.get('use_rag'))
        logger.info(f"Generated answer: {answer}")
        await message.reply(answer)

    except Exception as err:
        logger.error("Error while processing message", exc_info=err)
        await message.reply(f"Error while processing message:\n\n{err}")


async def start_bot():
    logger.info("Bot started")
    await dp.start_polling(bot)
