from multiprocessing import context
import os
from config import OPENAI_API_KEY, OPENAI_API_BASE

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, OPENAI_API_BASE
from dotenv import load_dotenv

# Настройка API
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE

# Инициализация эмбеддингов через прокси или напрямую
embeddings = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

# Загрузка FAISS-индекса с разрешением десериализации
vectorstore = FAISS.load_local(
    'faiss_index',
    embeddings,
    allow_dangerous_deserialization=True
)

# Настройка Retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 7})



# Инициализация чат-модели
chat_model = ChatOpenAI(
    model_name='gpt-4o-mini',
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    temperature=0
)

# Построение RAG-цепочки
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    #chain_type_kwargs={"prompt": prompt_template}
)


def is_admission_related_smart(question: str) -> bool:
    """Проверяет тематику через LLM"""
    check_prompt = f"""Определи, связан ли следующий вопрос с поступлением в университет/магистратуру.

Вопрос: "{question}"

Ответь только "ДА" если вопрос о:
- поступлении, документах, экзаменах
- выборе программ/кафедр
- процедурах подачи заявлений
- требованиях к поступающим
- сроках и дедлайнах
- учебных программах и специальностях

Ответь "НЕТ" если вопрос о погоде, развлечениях, общих темах, не связанных с поступлением.

Ответ:"""
    
    try:
        result = chat_model.invoke(check_prompt)
        return "ДА" in result.content.upper()
    except:
        return True  # в случае ошибки разрешаем вопрос


def answer_question(question: str) -> str:
    """
    Отвечает на вопрос с проверкой тематики через LLM, используя RAG.
    """
    # Проверяем тематику через LLM
    if not is_admission_related_smart(question):
        return """Я специализируюсь на вопросах поступления в магистратуру МФТИ. 

Могу помочь с:
• Подачей документов и сроками
• Вступительными испытаниями  
• Выбором кафедр и программ
• Требованиями к поступающим
• Процедурами зачисления

Задайте вопрос по этим темам!"""
    
    # Получаем релевантные документы (все, что вернул retriever)
    docs = retriever.get_relevant_documents(question)
    # Формируем контекст из всех документов
    context_parts = []
    for d in docs:
        context_parts.append(d.page_content)
    context = "\n".join(context_parts)
    # Формируем простой prompt без шаблона
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    result = chat_model.invoke(prompt)
    final = result.content.strip()
    if not final or final.lower().startswith("извините, я не смог") or final.lower().startswith("я не знаю"):
        return "Я не смогла найти подходящей информации, если вопрос очень важный — обратитесь к Юлии Синицыной."
    return final