__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_option_menu import option_menu
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key=st.secrets["api_key"]
embedding_openai = api_key

# embedding_openai = OpenAIEmbeddings()

CHROMA_DATA_PATH = 'embeddings_use_case_12_openai_semanticv2'
COLLECTION_NAME = 'embeddings_use_case_12_openai_semanticv2'

persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = persistent_client.get_collection(COLLECTION_NAME)

vectordb = Chroma(client=persistent_client, collection_name=COLLECTION_NAME,embedding_function=embedding_openai)


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=300,
    api_key=st.secrets["api_key"],
)

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ('system', 'You are a friendly assistant that answers questions on user inquiries in 300 tokens or less.'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad")       # need exact variable name
                                                                    # The agent prompt must have an `agent_scratchpad` key
    ]
)

def resume_retriever_tool(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    vector_store = Chroma.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        embedding=embedding_openai,
        metadatas=[{'use_case': 'resume_info'}] * len(chunks)  # Ensure metadatas is a list with the same length as documents
    )

    resume_retriever = vector_store.as_retriever(
    search_kwargs={
                   "filter": {"use_case": {"$eq": "resume_info"}}
                   }      # filter according to knowledgebase
    )

    resume_retriever_tool = create_retriever_tool(
        retriever=resume_retriever, 
        name='resume_search', 
        description='''Use this tool to parse the user's resume for details such as name, contact information, educational background, skillset (soft and technical skills), and job experiences. This tool can be used together with either of the two tools, but not both at the same time.
        First, it can be used in conjunction with the eskwelabs_bootcamp_info_search tool for queries related to an applicant's qualifications. For example, it can help assess if the user's skills and educational background are sufficient for specific programs like the Data Science Fellowship (DSF) or Data Analytics Bootcamp (DAB).
        Second, it can be used with the bootcamp_vs_alternatives_search tool to determine whether a user's skills and qualifications make a bootcamp or an alternative learning path a better option for them.
        '''
    )

    return resume_retriever_tool

def create_db_retriever_tools(vectordb):
    retriever_eskwelabs = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
                   "score_threshold": 0.5, 
                   "filter": {"use_case": {"$eq": "eskwelabs_faqs"}}
                   }      # filter according to knowledgebase
    )

    eskwelabs_bootcamp_info_search_tool = create_retriever_tool(
        retriever=retriever_eskwelabs,
        name="eskwelabs_bootcamp_info_search",
        description='''Use this tool to retrieve comprehensive information about Eskwelabs, specifically its Data Analytics Bootcamp (DAB) and Data Science Fellowship (DSF). 
        This tool can answer queries about Eskwelabs' tuition fees, equipment requirements, program duration, curriculum, enrollment processes, scholarship offers, and other specific details. 
        **Avoid using this tool for questions unrelated to Eskwelabs, such as general educational comparisons (e.g., comparing bootcamps with other learning methods), unrelated topics, or information about public figures or current events.**
        This tool is particularly useful when specific details about Eskwelabs are needed.
        Additionally, it can be used in conjunction with the resume_search tool for queries that assess an applicant's readiness or suitability for Eskwelabs programs based on their resume skills.
        '''
    )

    retriever_bootcamp_vs_alternatives = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
                   "score_threshold":  0.5, 
                   "filter": {"use_case": {"$eq": "bootcamp_vs_selfstudy"}}
                   }      # filter according to knowledgebase
    )

    bootcamp_vs_alternatives_search_tool = create_retriever_tool(
        retriever=retriever_bootcamp_vs_alternatives,
        name="bootcamp_vs_alternatives_search",
        description='''Use this tool to retrieve information about the pros and cons of bootcamps compared to other learning methods, such as online courses and academic institutions. 
        **Avoid using this tool for questions unrelated to educational comparisons, such as questions about public figures, unrelated topics, or specific bootcamp programs like Eskwelabs.** 
        This tool is not intended for queries about specific bootcamp details.
        Additionally, this tool can be used in conjunction with the resume_search tool for queries assessing whether a user's skills and qualifications make a bootcamp or an alternative learning path a better option for them.
        '''
    )

    return eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool

eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
tools = [eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]

agent=create_tool_calling_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

def process_chat(agent_executor, user_input, chat_history):
    # Convert Streamlit messages to the format expected by the agent
    formatted_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted_history.append(message)
        elif isinstance(message, AIMessage):
            formatted_history.append(message)
        else:
            # If the message is in the old format (dictionary), convert it
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))

    # Invoke the agent
    response = agent_executor.invoke(
        {
            'input': user_input,
            'chat_history': formatted_history
        }
    )
    return response['output']

def format_message(message):            # turns human/ai message instance to string (to be used for metadatas)
    if isinstance(message, HumanMessage):
        return f"Human: {message.content}"
    elif isinstance(message, AIMessage):
        return f"AI: {message.content}"
    
def parse_message(formatted_message):           # turns string into human/ai message type instance (to be fed to chat history)
    if formatted_message.startswith("Human: "):
        content = formatted_message[len("Human: "):]
        return HumanMessage(content=content)
    elif formatted_message.startswith("AI: "):
        content = formatted_message[len("AI: "):]
        return AIMessage(content=content)
    else:
        raise ValueError("Unknown message format")

def eskwelabs_chatbot():
    # Sidebar for file upload
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>Askwelabs</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Upload and Process Your Resume</h3>", unsafe_allow_html=True)
        docx_file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

        # Initialize session state variables
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'unique_id' not in st.session_state:
            st.session_state.unique_id = 0
        if 'chat_history_vector_store' not in st.session_state:
            st.session_state.chat_history_vector_store = None
        if 'agent_executor' not in st.session_state:
            st.session_state.agent_executor = agent_executor  # Use the global agent_executor
        if 'fed_chat_history' not in st.session_state:
            st.session_state.fed_chat_history = []    

        if docx_file is not None:
            file_details = {
                "Filename": docx_file.name,
                "FileType": docx_file.type,
                "FileSize": docx_file.size
        }
            st.write(file_details)

            # Save the uploaded file
            with open(docx_file.name, "wb") as f:
                f.write(docx_file.getbuffer())
    
            # Create the resume tool with the uploaded file
            resume_tool = resume_retriever_tool(docx_file.name)
    
            # Add the resume tool to a local copy of the tools list
            local_tools = tools.copy()
            if resume_tool not in local_tools:
                local_tools.append(resume_tool)
    
            st.success("File uploaded and processed. You can now ask questions about the document.")

        # Recreate the agent and agent_executor with the updated tools
            agent = create_tool_calling_agent(llm, local_tools, prompt)
            st.session_state.agent_executor = AgentExecutor(agent=agent, tools=local_tools, verbose=True)

    # Main area for chatbot interaction
    st.markdown("<h3 style='text-align: center;'>Chat with Askwelabs</h3>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("You: ")

    if user_input:
        # Similarity search in chat history vector store
        if st.session_state.chat_history_vector_store:
            results = st.session_state.chat_history_vector_store.similarity_search(
                query=user_input,
                k=4,
                filter={'use_case': 'chat_history'}
            )
            sequenced_chat_history = [(parse_message(result.metadata['msg_element']), result.metadata['msg_placement']) for result in results]
            sequenced_chat_history.sort(key=lambda pair: pair[1])
            st.session_state.fed_chat_history = [message[0] for message in sequenced_chat_history]

        # Process chat and get response
        response = process_chat(st.session_state.agent_executor, user_input, st.session_state.fed_chat_history)

        # Update messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update vector store
        formatted_human_message = format_message(HumanMessage(content=user_input))
        formatted_ai_message = format_message(AIMessage(content=response))

        if st.session_state.chat_history_vector_store:
            st.session_state.chat_history_vector_store.add_texts(
                texts=[user_input, response],
                ids=[str(st.session_state.unique_id), str(st.session_state.unique_id + 1)],
                metadatas=[
                    {'msg_element': formatted_human_message, 'msg_placement': str(st.session_state.unique_id), 'use_case': 'chat_history'},
                    {'msg_element': formatted_ai_message, 'msg_placement': str(st.session_state.unique_id + 1), 'use_case': 'chat_history'}
                ],
                embedding=embedding_openai
            )
            st.session_state.unique_id += 2
        else:
            st.session_state.chat_history_vector_store = Chroma.from_texts(
                texts=[user_input, response],
                ids=[str(st.session_state.unique_id), str(st.session_state.unique_id + 1)],
                metadatas=[
                    {'msg_element': formatted_human_message, 'msg_placement': str(st.session_state.unique_id), 'use_case': 'chat_history'},
                    {'msg_element': formatted_ai_message, 'msg_placement': str(st.session_state.unique_id + 1), 'use_case': 'chat_history'}
                ],
                embedding=embedding_openai
            )
            st.session_state.unique_id += 2

        # Force a rerun to update the chat display
        st.rerun()

# Function for Study Path
def study_path():
    
    st.markdown("<h3 style='text-align: center;'>Askwelabs Study Path Guidance</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Chat with Askwelabs to get personalized study path recommendations</h4>", unsafe_allow_html=True)
    
    ### Load ChromaDB ###
    CHROMA_DATA_PATH1 = 'embeddings_use_case_3_openai'
    COLLECTION_NAME2 = 'embeddings_use_case_3_openai'

    # chromadb openai wrapper for embedding layer
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=st.secrets["api_key"],
                    model_name="text-embedding-ada-002"
                )

    persistent_client2 = chromadb.PersistentClient(path=CHROMA_DATA_PATH1)

    study_path_collection = persistent_client2.get_collection(
        name=COLLECTION_NAME2,
        embedding_function=openai_ef)

    #### Filtering function from Part 1A ####
    def choosing_filter(filter_list):     # must pass a dictionary enclosed in a list
        default_filter = {'use_case': {'$eq': 'study_path'}}
        if len(filter_list) > 1:        # two or more filtering options chosen
            filter_list.append(default_filter)
            where_statement = {"$and": filter_list}
            return where_statement
        elif len(filter_list) == 1:     # one filtering option chosen only
            filter_list.append(default_filter)
            where_statement = {"$and": filter_list}
            return where_statement
        else: # no filtering (0 in list)
            return default_filter

    ### Final Chatbot Pipeline
    def chatbot_pipeline_input_embedding(query_text, n_results=3, filter_list=[]):
        embedding = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
        studypath_vectordb = Chroma(client=persistent_client2, collection_name=COLLECTION_NAME2, embedding_function=embedding)

        # Validate query text
        if not query_text.strip():
            return "Please enter a valid query text."

        where_statement = choosing_filter(filter_list)

        if where_statement is not None:
            studypath_retriever = studypath_vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                                "k": n_results,
                                "score_threshold": 0.3, # higher score means higher similarity needed to output results
                                "filter": where_statement
                                }      # filter according to knowledgebase
            )
        else:
            studypath_retriever = studypath_vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                                "k": n_results,
                                "score_threshold": 0.3, # higher score means higher similarity needed to output results
                                }
            )

        results = studypath_retriever.invoke(query_text)

        if results:
            st.write(f"Here are the top {len(results)} study path(s) that I recommend you to take:")
            for index, result in enumerate(results, 1):
                st.markdown(f"""
### {index}. {result.metadata['study_path']}

**Field:** {result.metadata['field']}

**Steps:** {result.metadata['steps']}

**Difficulty:** {result.metadata['difficulty_level']}

**Price:** {result.metadata['price']}

**Description:**  
{result.metadata['study_path_description']}

**URL:** [{result.metadata['link']}]({result.metadata['link']})

---
        """)
            return ""  # Return an empty string as we've already displayed the results
        else:
            return "No results were found for your query. It appears that none of our study paths match your search preferences."

    # Streamlit UI
    st.subheader("Enter your query and preferences")

    query_text = st.text_input("Enter your study path query:")

    # Sidebar filters
    st.sidebar.subheader("Filters")
    difficulty_level = st.sidebar.selectbox("Select difficulty level:", ["Any", "Beginner", "Intermediate", "Advanced"])
    field = st.sidebar.selectbox("Select field:", ["Any", "Data Scientist", "Data Analyst", "Both"])
    n_results = st.sidebar.slider("Number of results", min_value=1, max_value=5, value=3)

    # Mapping the selected field back to the corresponding filter value
    field_map = {
    "Any": "Any",
    "Data Scientist": "DS",
    "Data Analyst": "DA",
    "Both": "DS/DA"
    }

    field_value = field_map[field]

    if st.button("Get Recommendations"):
        if query_text:
            filter_list = []
            if difficulty_level != "Any":
                filter_list.append({'difficulty_level': {'$eq': difficulty_level}})
            if field_value != "Any":
                filter_list.append({'field': {'$eq': field_value}})
        
            # Handle the case where there is only one filter
            if len(filter_list) == 1:
                filter_dict = filter_list[0]
            elif len(filter_list) > 1:
                filter_dict = {'$and': filter_list}
            else:
                filter_dict = {}

            result = chatbot_pipeline_input_embedding(query_text, n_results, [filter_dict] if filter_dict else [])
            if result:
                st.write(result)
        else:
            st.warning("Please enter a query.")

    # Display current filters in the sidebar
    st.sidebar.subheader("Current Filters")
    st.sidebar.write(f"Difficulty Level: {difficulty_level}")
    st.sidebar.write(f"Field: {field}")
    st.sidebar.write(f"Number of Results: {n_results}")

def about():
    st.title("About Us")
    
    st.header("Askwelabs Chatbot")
    st.write(
        "The Askwelabs chatbot is designed for Eskwelabs, providing users with instant answers "
        "to inquiries about the company's offerings. Whether users are curious about course details, "
        "bootcamps, or other educational services, AskWelabs delivers accurate and helpful information "
        "to assist with their queries.")
        
    st.header("Study Path Page")
    st.write(
        "The Study Path Page offers personalized recommendations for study paths based on user inputs. "
        "By analyzing individual goals, skills, and preferences, this page suggests tailored learning paths "
        "and courses to help users achieve their educational and career objectives effectively."
    )

        # Meet the Team Section
    st.header("Meet the Team")
    st.write("""
        We are a passionate group of learners from the Data Science Cohort 13 with a shared vision: 
        to empower future data scientists and analysts to reach their full potential. 
        Our team brings together diverse expertise from the fields of engineering and technology to create 
        a unique learning experience for aspiring professionals.
    """)

        # Maybe individual pic and description of each person and their role in this 

        # Our Motivation Section
    st.header("Our Motivation")
    st.write("""
        The journey to becoming a data scientist or data analyst is both exciting and challenging. 
        We understand the significance of thorough preparation and the impact it can have on your success 
        in Eskwelab's bootcamps. Our motivation stems from the desire to bridge the gap between where you are 
        now and where you want to be, ensuring that you are well-prepared to thrive in this rigorous learning environment.
    """)

        # Why We Created This App Section
    st.header("Why We Created This App")
    st.write("""
        As a future data scientist or analyst, you aim to maximize your experience in the Eskwelab's bootcamps. 
        We created this virtual coach to guide you every step of the way, providing you with the resources, 
        tools, and personalized guidance necessary to excel. Our goal is to help you make the most of your time, 
        ensuring that you gain the skills, knowledge, and confidence needed to succeed in the bootcamp and beyond.
    """)


# Main function to handle navigation
def main():
    selected = option_menu(
        menu_title=None,  # No title
        options=["Home", "Study Path", "About"],  # Menu options
        icons=["house", "book", "list-task"],  # Icons for the menu options
        menu_icon="cast",  # Icon for the menu
        default_index=0,  # Default selected option
        orientation="horizontal",  # Menu orientation

        styles={
        "icon": {"color": "white"}, 
        "nav-link": {"--hover-color": "#ffff"},
        "nav-link-selected": {"background-color": "#0e7d74"},
        }
    )

    if selected == "Home":
        eskwelabs_chatbot()  # Call the Eskwelabs Chatbot function
    elif selected == "Study Path":
        study_path()
    elif selected == "About":
        about()

if __name__ == '__main__':
    main()
