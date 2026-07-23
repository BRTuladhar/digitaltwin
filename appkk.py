import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import requests
import random

#---------------------------------------------------
# Setup
#---------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()

#---------------------------------------------------
# Documents 
#---------------------------------------------------
document_overview = """ 
Kirill runs SuperDataScience, an online platform helping people learn data science and AI. 
He's based in Brisbane, Australia. He has a Master's in Applied Finance and Accounting 
from the University of Queensland which he finished in 2012. 

He's been teaching on Udemy since 2015 and created courses like Machine Learning A-Z and 
GenAI & LLMs A-Z with his business partner Hadelin de Ponteves. He also hosted the 
Super Data Science Podcast between 2016-2021, then handed over the reins to Jon Krohn.

Other career history:
2014-2015: Data Scientist at Sunsuper, an Australian superannuation fund
2012-2014: Analyst at Deloitte Australia

What drives him: He loves building courses and educational content. It's rewarding to help 
people learn through journeys with lots of "Aha" moments. He believes learning shouldn't be lonely — 
members often struggle with motivation, so building a supportive community matters to him.

His approach: Practical and accessible. Data science can sound intimidating, but it doesn't have to be. 
He'd rather help someone build something real than get lost in theory.

Communication style: Direct, friendly, and encouraging. Happy to share what he's learned, 
but also still learning — this field moves fast.

Additional info:
- In 2001 Kirill was actively playing chess
- Kirill enjoys cooking and often experiments with new recipes in his free time
- Kirill has a fondness for pineapple on pizza and often debates its merits with friends.\
The reason Kirill thinks pineapple on pizza is delicious is because of the sweet and \
savory contrast it provides, which he finds delightful to the palate. In addition to \
the flavor, he also enjoys the juicy texture that pineapple adds to the pizza, making\
each bite more enjoyable. The combination of the tangy pineapple with the savory \
creates a unique taste that cannot be achieved with other toppings, which is why \
Kirill is a fan of pineapple on pizza.
"""

document_education = """The University of Queensland logo
The University of Queensland

Master of Commerce, Applied Finance and Professional Accounting

2010 – 2012

Grade: 6.4 / 7.0 (Dean's Commendations for Academic Excellence)

Activities and societies: Business Economics and Law faculty

High achieving Masters student at one of the World's top 50 Universities. Took deep dives into the fields of Accounting, Business Statistics, Economics, and Finance. Coursework projects involved hypothesis testing, market analysis, and financial forecasting to name but a few. 

Moscow Institute of Physics and Technology (State University) (MIPT) logo
Moscow Institute of Physics and Technology (State University) (MIPT)

Bachelor of Science, Applied Physics and Mathematics

2006 – 2010

Grade: 4.9 / 5.0 (With Honors)

Activities and societies: DGAP, FOPF, ZFTSH

Top student at the one of the most demanding Mathematics and Physics Universities in the World. Applied statistical techniques and predictive modelling to test hypotheses and facilitate experimental research. Involved in a number of research projects in the fields of IT, Laser Physics, and Applied Mathematics. """

document_professional_experience = """SuperDataScience logo
Founder & CEO

SuperDataScience

Jun 2015 - Present · 10 yrs 9 mos

Global

SuperDataScience is an AI education company helping individuals and enterprises thrive in the AI age.
• For enterprises: we deliver high-production AI Literacy programs, including our enterprise-ready AI Literacy courses for company-wide rollout.
• For individuals: we offer online courses, a leading AI industry podcast, a global learning community, and an 8-week AI Engineer Bootcamp that takes professionals from GenAI fundamentals to production-grade Agentic AI deployment in AWS.


CloudWolf logo
Co-Founder

CloudWolf · Full-time

Jan 2023 - Present · 3 yrs 2 mos

Global · Hybrid

CloudWolf is an AWS education company helping professionals and teams master cloud skills and prepare for AWS exams.
• For enterprises: we deliver production-grade AWS training programs focused on scalable, secure, and cost-optimized AI infrastructure.
• For individuals: we offer hands-on AWS courses, certification prep, and real-world projects that teach how to build, deploy, and operate AI systems on AWS with confidence


Udemy logo
Instructor

Udemy

Jun 2014 - Present · 11 yrs 9 mos

San Francisco Bay Area

With over 3M students world-wide, Kirill is passionate about education and his Personal Mission is to "Make The Complex Simple". In his courses he breaks down complex topics like Artificial Ingelligence, GenAI, Machine Learning, Data Science into easily digestible tutorials. The goal is to eliminate barriers to entry so students can enhance their professional careers with the power of Data.


Super Data Science Podcast logo
Founder

Super Data Science Podcast

Sep 2016 - Present · 9 yrs 6 mos

Global

For Listeners: The SuperDataScience podcast brings you the most inspiring AI Engineers, Data Scientists, Researchers and other professionals. AI & Data are growing and this podcast will help you keep abreast with the latest technology in this space.

For Industry Experts: If you have a message to share with our AI & Data community and would like to be a guest on our show, please contact us at podcast@superdatascience.com


DataScienceGO logo
Founder

DataScienceGO

May 2017 - Dec 2021 · 4 yrs 8 mos

Global

DataScienceGO is the crossroad where potential meets guidance. The tipping point to your Data Science career. An event created for data powered minds. Where experts, mentors and friends come to enlighten, click and inspire each other and skyrocket their careers. Check out our in-person and Virtual Conferences. Are you in?


Sunsuper logo
Data Scientist

Sunsuper

Jul 2014 - Jun 2015 · 1 yr

Greater Brisbane Area

Operational Efficiency: Delivered a response forecasting and resource allocation model for the National 'Supermatch' campaign. Cross-departmental implementation of the model significantly uplifted operational efficiency reducing processing times by 93% (from 40 days to 3 days). This model was a key driver of an unprecedented customer experience while successfully delivering over $142M increase in FUM.

Customer Segmentation: Kirill developed multiple predictive models to drastically increase ROI of campaigns targeted at Churn, Reactiviation, Email Collection, to name but a few. Models were delivered and applied in both the DM and eDM spaces uplifting campaign effectiveness by 15% to 25% on average. Sophisticated Data Science in the customer space played a key role in Sunsuper winning the Chant West 'Best Fund: Member Services' award.

Tableau Dashboards: Kirill led business-wide roll-out of Tableau Server and created multiple Tableau MIS dashboards facilitating granular segmentation of the member base. The results have been instrumental in the delivery of tailored communications, ensuring that members of the fund have a truly unique experience. The dashboards have been presented to 10+ Superannuation Tender Consultants (like Rice Warner, Chant West) and are deemed to have raised industry standards of Customer Analytics.

Thumbnail for AIST CMSF Event 2015
AIST CMSF Event 2015

Coverage of the 2015 AIST Conference of Major Superannuation Funds


Deloitte Decision Science & Analytics logo
Analyst

Deloitte Decision Science & Analytics

Jul 2012 - Jul 2014 · 2 yrs 1 mo

Greater Brisbane Area

Global Mining Services Provider, Business Analytics: Lead a team of four on a quantitative risk analytics and mitigation project for customer contracts worth more than $18M (part of a $100M+ business integration). Developed KPIs and business processes, increasing efficiency of risk mitigation procedures by 110%. Major skills transfer by coaching and training business analysts and other staff. Kirill performed a national role, coordinated work across 3 sites, and presented to executive officers.

Australian Retail Medical Facility, Predictive Modelling: $42M profit (EV/EBITDA multiple of 6) achieved on sale of the client's business through combined data science and financial analytics. Kirill supervised a team of three and was in charge of developing a predictive statistical model in SQL which drove financial forecasts. Examples of techniques used in the model include: customer profiling, geo-demographic market segmentation, multiple linear regression, and principal component analysis.

Australian Public Sector, Visual Analytics: Kirill developed MIS dashboards in Tableau which drove optimisation of analytical solutions resulting in cost savings of over $19M. Data visualisation helped achieve cost reduction with minimal impact on stakeholders. Kirill was also responsible for data gathering, interrogation, and manipulation, as well as supporting the client post project delivery."""

#---------------------------------------------------
# Chunking Function
#---------------------------------------------------
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    BOUNDARIES = ["\n\n", "\n", ". ", "? ", "! ", " "]
    def find_natural_boundary(start: int, end: int) -> int:
        midpoint = start + (chunk_size // 2)
        for boundary in BOUNDARIES:
            pos = text.rfind(boundary, midpoint, end)
            if pos != -1:
                return pos + len(boundary)
        return end

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_natural_boundary(start, end)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

#---------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
#---------------------------------------------------
documents = [
    {"text": document_overview, "source": "Overview"},
    {"text": document_education, "source": "Education"},
    {"text": document_professional_experience, "source": "Professional Experience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    #Prepare the lists
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=300, overlap=30)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]
    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#Print for logs    
print(f"Created {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}):")
    print(chunk)
    print()

#Generate embeddings
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

#Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

#intialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")
#Alternative: intialize ChromaDB client (in-memory storage)
#chroma_client = chromadb.Client()

#Get or Create + Empty the collection before adding new data (for testing purposes)
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

#Adding data to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings, 
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())

#---------------------------------------------------
# Tools
#---------------------------------------------------
tools = []

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

#Create send_notification function
def send_notification(message: str):
    if pushover_user is None or pushover_token is None: #Handling of potential missing crednetials
        return "Notification failed: Pushover not configured."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

#Describe Pushover as an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real Kirill. Use this when: \
            1) Someone wants to get in touch, hire, or collaborate\
            - ask for their name and contact details first, then send notification to Kirill with the name and contact details. \
            2) You don't know the answer to a question about Kirill - send AUTOMATICALLY without asking, include the question so he can add this info later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The notificaiton message to send to the user's device"}
        },
        "required": ["message"]
    }
}

#Add Pushover to the list of tools for the LLM
tools.append({"type":"function", "function":send_notification_function})

#Simulates rolling a single six-sided die
def dice_roll():
    result = random.randint(1,6)
    return result

#Describe function for the LLM
roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions, or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

#Add function to list of tools of LLM
tools.append({"type":"function", "function":roll_dice_function})

#---------------------------------------------------
# Tool Handler
#---------------------------------------------------
def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #print(f"Calling function {function_name}") #For future debugging ;)

        #Route to the appropriate function based on function_name
        if function_name == "send_notification":
            content = send_notification(args["message"])
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        #elif function_name == "insert_function_name_3":
        #   content = insert_function_name_3(args["message"])
        #....
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }
        tool_results.append(tool_call_result)
    
    return tool_results

#---------------------------------------------------
# System Message
#---------------------------------------------------
system_message = """You are a digital twin of Kirill Eremenko. When people talk to you, 
you respond AS Kirill — in first person, using his voice, personality, and knowledge.

Important: do not make things up. If you don't know an answer, say you don't know. 
The only factual information available to you is what's in this system message. 
You cannot get any more facts about Kirill from the internet or make them up.

IMPORTANT: Whenever you don't know something about Kirill, 
ALWAYS use the send_notification tool to alert the real Kirill — do this automatically without asking the user.
"""

#---------------------------------------------------
# Main Response Function
#---------------------------------------------------
def respond_ai(message, history):
    #RAG: Embed the query using the same model we used for the chunks to ensure compatibility
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )
    query_embedding = response.data[0].embedding

    #RAG: Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    #RAG: Stitch retrieved chunks together to create the context for the response
    context = "\n---\n".join(results["documents"][0])
    
    #Print logs for debugging
    print("\n===========================\n")
    print(f"User message:\n{message}\n")
    print("***Retrieved Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print("-------------------")
        print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")

    #Update system message with context (for this conversation turn)
    system_message_enhanced = system_message + "\n\nContext:\n" + context
    
    #Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]
    
    #Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools   
    )
    message = response.choices[0].message

    #Check if model wants to call a tool
    while message.tool_calls:
        pprint (message.tool_calls)
       
        tool_result = handle_tool_call(message.tool_calls) #whole list of tool calls on purpose
        messages.append(message)
        messages.extend(tool_result) #Changed from append() to extend() when we switched to multiple tool call handling
    
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        #Note: maybe consider adding protection from infinite consecutive tool calling

    return(message.content)

#---------------------------------------------------
# Launch Gradio
#---------------------------------------------------
gr.ChatInterface(
    fn=respond_ai,
    title="Kirill's Digital Twin",
    chatbot=gr.Chatbot(avatar_images=(None, "kirill.jpeg")),
    description="Chat with an AI version of Kirill Eremenko. Ask about his experience, projects, or just say hi!",
    examples=["What's your background?", "AI Engineering experience", "Do you like pineapple on pizza?"]
).launch()