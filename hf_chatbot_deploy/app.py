"""Rafiqa (رفيقة) - Pre-Marital Health Assistant Gradio app."""
import os
import sys
import uuid
import tempfile
import shutil
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.emergency import check_emergency
from src.admin_store import (
    log_conversation, log_emergency, log_upload, get_full_admin_data,
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ─── Configuration ────────────────────────────────
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "pre-marital-health-assistant")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "pre_marital_health_assistant")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY_TURNS = 12
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "Chat-Bot-1")

# ─── Initialize Components ───────────────────────
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

chatModel = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    """Format retrieved documents as a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt_template
    | chatModel
    | StrOutputParser()
)


# ─── Chat Logic ──────────────────────────────────
def build_conversation_context(state, max_turns=MAX_HISTORY_TURNS):
    """Build a conversation context string from recent turns."""
    if not state:
        return ""
    turns = state[-max_turns:]
    return "\n\n".join(
        [f"المستخدم: {u}\nرفيقة: {a}" for u, a in turns]
    )


def chatbot_fn(user_message, chat_history, state):
    """Main chatbot handler with emergency detection and RAG."""
    chat_history = chat_history or []
    state = state or []

    # Generate session ID on first message
    session_id = None
    if state and len(state) > 0 and isinstance(state[0], str):
        session_id = state[0]
        actual_turns = state[1:] if len(state) > 1 else []
    else:
        session_id = str(uuid.uuid4())[:8]
        actual_turns = state if state else []

    now_time = datetime.now().strftime('%H:%M')
    is_emergency = False
    emergency_symptom = ""

    # 1. Emergency check first
    emergency_check = check_emergency(user_message)
    if emergency_check["is_emergency"]:
        assistant_reply = emergency_check["response"]
        is_emergency = True
        emergency_symptom = "، ".join(emergency_check.get("detected_symptoms", ["أعراض خطيرة"]))
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        actual_turns.append((user_message, assistant_reply))

        # Log emergency alert
        conv_num = len(actual_turns)
        user_name = f"مستخدم #{session_id}"
        try:
            log_emergency(user_name, emergency_symptom, user_message, session_id)
        except Exception as e:
            print(f"[AdminStore] Error logging emergency: {e}", file=sys.stderr)

        # Log conversation
        try:
            msgs = []
            for i, (u, a) in enumerate(actual_turns):
                msgs.append({"id": i*2+1, "role": "user", "content": u, "time": now_time})
                msgs.append({"id": i*2+2, "role": "assistant", "content": a, "time": now_time})
            log_conversation(session_id, user_name, msgs,
                           is_emergency=True, emergency_symptom=emergency_symptom)
        except Exception as e:
            print(f"[AdminStore] Error logging conversation: {e}", file=sys.stderr)

        new_state = [session_id] + actual_turns
        return chat_history, new_state

    # 2. Build conversation context
    conversation_context = build_conversation_context(actual_turns)
    if conversation_context:
        combined_input = (
            f"سياق المحادثة السابقة:\n{conversation_context}\n\n"
            f"السؤال الحالي: {user_message}"
        )
    else:
        combined_input = user_message

    # 3. Get RAG response
    try:
        result = rag_chain.invoke(combined_input)
        assistant_reply = result if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"Error in chatbot_fn: {e}", file=sys.stderr)
        assistant_reply = (
            "عذرًا يا عزيزتي، حصل خطأ أثناء معالجة سؤالك. "
            "ممكن تحاولي تاني؟ 🤍"
        )

    # 4. Update history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_reply})
    actual_turns.append((user_message, assistant_reply))

    if len(actual_turns) > 40:
        actual_turns = actual_turns[-40:]

    # 5. Log to admin store
    try:
        user_name = f"مستخدم #{session_id}"
        msgs = []
        for i, (u, a) in enumerate(actual_turns):
            msgs.append({"id": i*2+1, "role": "user", "content": u, "time": now_time})
            msgs.append({"id": i*2+2, "role": "assistant", "content": a, "time": now_time})
        log_conversation(session_id, user_name, msgs)
    except Exception as e:
        print(f"[AdminStore] Error logging conversation: {e}", file=sys.stderr)

    new_state = [session_id] + actual_turns
    return chat_history, new_state


def reset_chat():
    """Reset chat history and state."""
    return [], []


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def upload_and_process(file_path, admin_key):
    """Accept a PDF/TXT/MD file, embed its chunks and upsert them into Pinecone."""
    # Auth check
    if not admin_key or admin_key.strip() != ADMIN_API_KEY:
        return "❌ مفتاح الأدمن غير صحيح"

    if file_path is None:
        return "❌ لم يتم اختيار ملف"

    try:
        # Handle both string path and dict from Gradio API
        if isinstance(file_path, dict):
            actual_path = file_path.get("path", file_path.get("name", ""))
        else:
            actual_path = str(file_path)

        print(f"[Upload] Received file_path type={type(file_path)}, actual_path={actual_path}", file=sys.stderr)

        from src.helper import (
            load_pdf_file, load_text_file, filter_to_minimal_docs, text_split,
        )

        file_name = os.path.basename(actual_path)
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            return f"❌ نوع الملف غير مدعوم ({ext}). الأنواع المسموحة: PDF, TXT, MD"

        if not os.path.exists(actual_path):
            return f"❌ الملف غير موجود: {actual_path}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = os.path.join(tmp_dir, file_name)
            shutil.copy2(actual_path, dest)

            if ext == ".pdf":
                docs = load_pdf_file(tmp_dir)
            else:
                docs = load_text_file(dest)

            if not docs:
                return f"❌ لم يتم العثور على محتوى في: {file_name}"

            filtered = filter_to_minimal_docs(docs)
            chunks = text_split(filtered, chunk_size=500, chunk_overlap=50)

            PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=PINECONE_INDEX,
                namespace=PINECONE_NAMESPACE,
            )

            # Log upload to admin store
            try:
                file_size_mb = f"{os.path.getsize(actual_path) / (1024*1024):.2f} MB"
            except Exception:
                file_size_mb = ""
            try:
                log_upload(file_name, file_size_mb, len(chunks), len(docs))
            except Exception as e:
                print(f"[AdminStore] Error logging upload: {e}", file=sys.stderr)

            return (
                f"✅ تم رفع الملف بنجاح!\n"
                f"📄 اسم الملف: {file_name}\n"
                f"📃 عدد الصفحات: {len(docs)}\n"
                f"✂️ عدد الأجزاء (Chunks): {len(chunks)}\n"
                f"🌲 Pinecone: {PINECONE_INDEX} / {PINECONE_NAMESPACE}"
            )
    except Exception as e:
        return f"❌ خطأ أثناء المعالجة: {str(e)}"


# ─── Custom CSS ───────────────────────────────────
custom_css = """
/* Global RTL and Arabic Font */
.gradio-container {
    direction: rtl !important;
    font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif !important;
}

/* Import Cairo Arabic font */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');

/* Header styling */
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #06B6D4, #8B5CF6, #EC4899);
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
}

.app-header h1 {
    font-size: 2.2em;
    margin-bottom: 5px;
}

.app-header p {
    font-size: 1.1em;
    opacity: 0.9;
}

/* Chat container */
.chatbot {
    direction: rtl !important;
    font-family: 'Cairo', sans-serif !important;
    min-height: 500px !important;
    border-radius: 16px !important;
    border: 2px solid #CFFAFE !important;
}

/* User message bubble */
.message.user {
    background: linear-gradient(135deg, #06B6D4, #0891B2) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    direction: rtl !important;
    text-align: right !important;
}

/* Bot message bubble */
.message.bot {
    background: linear-gradient(135deg, #F0FDFA, #CCFBF1) !important;
    color: #111111 !important;
    border-radius: 18px 18px 18px 4px !important;
    direction: rtl !important;
    text-align: right !important;
    border: 1px solid #99F6E4 !important;
}

/* Force ALL text in chatbot to be black */
[data-testid="chatbot"] *,
.chatbot *,
.chatbot .message *,
.chatbot .markdown-text *,
.chatbot p, .chatbot span, .chatbot div,
.chatbot li, .chatbot strong, .chatbot em,
.chatbot h1, .chatbot h2, .chatbot h3 {
    color: #111111 !important;
}

/* Keep user message text white */
[data-testid="chatbot"] .user *,
.chatbot .message.user *,
.message.user *,
.message.user p,
.message.user span,
.message.user div {
    color: white !important;
}

/* Input textbox */
.textbox textarea {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    border: 2px solid #CFFAFE !important;
    padding: 12px 16px !important;
}

.textbox textarea:focus {
    border-color: #06B6D4 !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2) !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #06B6D4, #0891B2) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
}

button.secondary {
    background: #F0FDFA !important;
    border: 2px solid #CFFAFE !important;
    border-radius: 12px !important;
    color: #0891B2 !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 600 !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 15px;
    color: #9CA3AF;
    font-size: 0.9em;
    margin-top: 10px;
}

/* Disclaimer banner */
.disclaimer {
    background: #FEF3C7;
    border: 1px solid #F59E0B;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 15px;
    text-align: center;
    color: #92400E;
    font-size: 0.95em;
}
"""

# ─── Gradio UI ────────────────────────────────────
with gr.Blocks(
    title="رفيقة — مساعد الصحة قبل الزواج 💍",
) as demo:

    # Header
    gr.HTML("""
    <div class="app-header">
        <h1>💍 رفيقة</h1>
        <p>مساعدتك الذكية للتوعية الصحية قبل الزواج — صديقة واعية مش طبيبة</p>
    </div>
    """)

    # Disclaimer
    gr.HTML("""
    <div class="disclaimer">
        ⚠️ رفيقة مساعدة إرشادية فقط ولا تُغني عن استشارة الطبيب المختص.
        في حالة الطوارئ أو الخطر المباشر، تواصلي مع الطوارئ فورًا.
    </div>
    """)

    # Chat interface
    chatbot = gr.Chatbot(
        label="رفيقة",
        value=[],
        height=500,
    )

    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="اكتب/ي سؤالك الصحي قبل الزواج هنا... 🤍",
            scale=6,
            container=False,
        )
        submit_btn = gr.Button(
            "إرسال 💬",
            variant="primary",
            scale=1,
        )

    with gr.Row():
        clear_btn = gr.Button("🗑️ محادثة جديدة", variant="secondary")

    # ── Admin: Knowledge Base Upload ────────────────
    with gr.Accordion("⚙️ إدارة قاعدة المعرفة (Admin)", open=False):
        gr.HTML(
            "<p style='color:#666;font-size:14px;text-align:right;'>"
            "رفع ملفات (PDF, TXT, MD) جديدة لإضافتها إلى قاعدة معرفة رفيقة قبل الزواج"
            "</p>"
        )
        admin_key = gr.Textbox(
            label="مفتاح الأدمن",
            type="password",
            placeholder="أدخلي مفتاح الأدمن",
        )
        upload_file = gr.File(
            label="📎 اختر ملف (PDF, TXT, MD)",
            type="filepath",
        )
        upload_btn = gr.Button("📤 رفع ومعالجة الملف", variant="primary")
        upload_output = gr.Textbox(
            label="نتيجة العملية",
            interactive=False,
            lines=5,
            placeholder="ستظهر هنا نتيجة الرفع والمعالجة...",
        )

        upload_btn.click(
            fn=upload_and_process,
            inputs=[upload_file, admin_key],
            outputs=upload_output,
            api_name="upload_document",
        )

    # Event handlers
    msg.submit(
        chatbot_fn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state],
    ).then(lambda: gr.update(value=""), [], [msg])

    submit_btn.click(
        chatbot_fn,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state],
    ).then(lambda: gr.update(value=""), [], [msg])

    clear_btn.click(fn=reset_chat, outputs=[chatbot, state])

    # ── Admin API: Fetch dashboard data ────────────
    admin_api_key_input = gr.Textbox(label="Admin Key", visible=False)
    admin_api_output = gr.Textbox(label="Admin Data", visible=False)
    admin_api_btn = gr.Button("Fetch Admin Data", visible=False)
    admin_api_btn.click(
        fn=lambda api_key: get_full_admin_data(api_key, ADMIN_API_KEY),
        inputs=[admin_api_key_input],
        outputs=[admin_api_output],
        api_name="get_admin_data",
    )

    # Footer
    gr.HTML("""
    <div class="app-footer">
        <p>رفيقة — صُنعت بحب لدعم كل مقدم/ة على الزواج 🤍</p>
        <p style="font-size: 0.8em;">هذا التطبيق للأغراض الإرشادية فقط وليس بديلاً عن المشورة الطبية المتخصصة</p>
    </div>
    """)

    # Example questions
    gr.Examples(
        examples=[
            "ما هي أهم الفحوصات الطبية قبل الزواج؟",
            "أنا قلقة من نتائج التحاليل، أتصرف إزاي؟",
            "هل العمر بيأثر على الخصوبة؟",
            "إزاي أحسن نمط حياتي قبل الزواج؟",
            "إيه الفرق بين فحص CBC وفصيلة الدم وRh؟",
            "متى أحتاج أراجع طبيب نساء؟",
            "متى أحتاج أراجع طبيب ذكورة؟",
            "كيف أجهز نفسي نفسيًا قبل الزواج؟",
        ],
        inputs=msg,
        label="أسئلة شائعة 💡",
    )


# ─── Launch ───────────────────────────────────────
if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        css=custom_css,
    )
