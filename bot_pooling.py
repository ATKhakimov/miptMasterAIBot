from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from rag_bot import answer_question
import os
from dotenv import load_dotenv
from config import TELEGRAM_BOT_USERNAME as BOT_USERNAME

async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    # удалим упоминание из текста
    cleaned = text.replace(f"@{BOT_USERNAME}", "").strip()
    if not cleaned:
        return  # если после упоминания нет текста, выходим
    # необязательно: показать «печатает…»
    await update.message.chat.send_action(action="typing")
    # получаем и отправляем ответ
    reply = answer_question(cleaned)
    await update.message.reply_text(reply, reply_to_message_id=update.message.message_id)

def main():
    load_dotenv("keys.env")  # убедитесь, что у вас есть этот файл с переменными окружения
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # ловим любые тексты, содержащие @BOT_USERNAME
    mention_filter = filters.TEXT & filters.Regex(fr"@{BOT_USERNAME}")
    app.add_handler(MessageHandler(mention_filter, handle_mention))

    print("Бот запущен в polling-режиме (отвечает на упоминания).")
    app.run_polling()

if __name__ == "__main__":
    main()
