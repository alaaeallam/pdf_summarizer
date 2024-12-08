import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langdetect import detect
import warnings

# Suppress specific warnings about empty content
warnings.filterwarnings("ignore", message="Warning: Empty content on page")

# Initialize LLM
base_url = "http://localhost:11434"  # Replace with public URL if deployed
model = "llama3.2:3b"
llm = ChatOllama(base_url=base_url, model=model)

# Streamlit UI
st.title("Multilingual PDF Assistant (English/Arabic)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file locally
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load the PDF using PyMuPDFLoader
    try:
        loader = PyMuPDFLoader("uploaded_file.pdf")
        docs = loader.load()

        # Combine text from all pages, skipping empty ones
        def format_docs(docs):
            valid_pages = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
            return "\n\n".join(valid_pages) if valid_pages else None

        context = format_docs(docs)
        if not context:
            st.error("The uploaded PDF contains no readable content. Please upload a valid document.")
            st.stop()

        st.subheader("Extracted Text")
        st.text_area("Text from PDF", context, height=300)
    except Exception as e:
        st.error(f"Error loading the PDF: {str(e)}")
        st.stop()

    # Detect Language
    try:
        language = detect(context[:500])  # Detect language using a snippet of the text
        lang_display = "English" if language == "en" else "Arabic" if language == "ar" else "Other"
    except Exception:
        language = "unknown"
        lang_display = "Unknown"
    st.write(f"Detected Language: {lang_display}")

    # Summarization Section
    st.subheader("Summarize the Document")
    words = st.number_input("Number of words in summary", min_value=10, max_value=500, value=50, step=10)

    if st.button("Generate Summary"):
        try:
            if language == "ar":
                system = SystemMessagePromptTemplate.from_template("""أنت مساعد مالي متخصص يلخص المستندات المالية.
                                                                     قدم رؤى رئيسية واتجاهات ومقاييس من النص المعطى.""")
                prompt = """لخص المستند التالي لرؤى مالية في {words} كلمة:
                            ### النص:
                            {context}
                            
                            ### الملخص المالي:"""
            else:  # Default to English
                system = SystemMessagePromptTemplate.from_template("""You are a financial analyst assistant specializing in summarizing financial documents. 
                                                                     Provide key insights, trends, and metrics from the context provided.""")
                prompt = """Summarize the following document for financial insights in {words} words:
                            ### Context:
                            {context}
                            
                            ### Financial Summary:"""

            prompt = HumanMessagePromptTemplate.from_template(prompt)
            messages = [system, prompt]
            template = ChatPromptTemplate(messages)
            summary_chain = template | llm | StrOutputParser()

            # Validate inputs
            if not context.strip():
                st.error("Context is empty. Cannot generate a summary.")
                st.stop()

            # Generate summary
            response = summary_chain.invoke({'context': context, 'words': words})
            st.subheader("Summary")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")

    # Q&A Section
    st.subheader("Ask Questions about the Document")
    question = st.text_input("Enter your question:" if language == "en" else "أدخل سؤالك:")

    if st.button("Get Answer"):
        try:
            if language == "ar":
                system = SystemMessagePromptTemplate.from_template("""أنت مساعد مالي يجيب على الأسئلة بناءً على النص المتوفر.
                                                                     إذا لم تكن الإجابة موجودة في النص، فقل 'لا أعرف'.""")
                prompt = """أجب على السؤال التالي بناءً على النص:
                            ### النص:
                            {context}
                            
                            ### السؤال:
                            {question}
                            
                            ### الإجابة:"""
            else:  # Default to English
                system = SystemMessagePromptTemplate.from_template("""You are a financial assistant answering questions based on the provided document context.
                                                                     If the answer is not in the context, say 'I don't know'.""")
                prompt = """Answer the following question based on the context:
                            ### Context:
                            {context}
                            
                            ### Question:
                            {question}
                            
                            ### Answer:"""

            prompt = HumanMessagePromptTemplate.from_template(prompt)
            messages = [system, prompt]
            qna_chain = ChatPromptTemplate(messages) | llm | StrOutputParser()

            # Validate inputs
            if not context.strip():
                st.error("Context is empty. Cannot answer questions.")
                st.stop()
            if not question.strip():
                st.error("Question is empty. Please enter a valid question.")
                st.stop()

            # Generate answer
            answer = qna_chain.invoke({'context': context, 'question': question})
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
