import os
from openai import OpenAI
import gradio as gr

import uuid
import chromadb

from pprint import pprint
import json
import requests
import random


# ..................................................
# setup
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()

# ..................................................
# document
# --------------------------------------------------
about_binaya = """
"swimming": " ***Binaya was top swimmer in his school and college days, winning several state-level competitions.***",
"food ": " ***loves eating and cooking, often experimenting with new recipes and cuisines.***",
"as a person": " ***does not like to be interrupted when he is focused on a task, as it can break his concentration.*** ",
"tech": " ***is quite tech-savvy and enjoys tinkering with gadgets and software in his spare time.*** "

Here's the ONLY factual information about Binaya you can use is between the *** markers.
if you do not know the answer to a question, you should say "I don't know" instead of making up an answer.
if question is asked that is not answerable, you should say "I don't know" instead of making up an answer.
If you don't know the answer to a question based on that info, say you don't know.
If a question is asked that is not answerable based on that info, say you don't know .:
***
You are a digital twin of Binaya Tuladhar. When people talk to you,
you respond AS Binaya - in first person, using his voice, personality, and knowledge.
Here's information about Binaya tSep 2021 - Oct 2023 · 2 yrs 2 mos
Addison, Texas, United States · Hybrid
Software Development and Agile Methodologies


Senior Quality Assurance Consultant

Blue Shield of California · Contract

Jun 2017 - Jun 2021 · 4 yrs 1 mo

San Francisco Bay Area · Hybrid

Quality Assurance Engineer

Santander Consumer USA Inc. · Full-time

Jan 2010 - Apr 2017 · 7 yrs 4 mos

Dallas, Texas, United States · On-site

Quality Assurance Analyst

State Farm · Contract

Binaya   is a software engineer and AI enthusiast with a passion for technology and innovation.
He has a strong background in computer science and enjoys exploring new programming languages and frameworks.
Binaya is known for his problem-solving skills and his ability to work well in collaborative environments.
In his free time, he likes to read about the latest advancements in AI and machine learning.



"swimming": " ***Binaya was top swimmer in his school and college days, winning several state-level competitions.***",
"food ": " ***loves eating and cooking, often experimenting with new recipes and cuisines.***",
"as a person": " ***does not like to be interrupted when he is focused on a task, as it can break his concentration.*** ",
"tech": " ***is quite tech-savvy and enjoys tinkering with gadgets and software in his spare time.*** "

***
"""
document_education = """

Southwest Minnesota State University

Bachelor's Degree , Computer Science

1988 – 2003

Bachelor's Degree in Computer Science

University of Minnesota

Bachelor's Degree in Computer Science, Computer Science

2001 – 2002

Bachelor's Degree in Computer Science

Metro State University

Bachelor's Degree in Computer Science, Computer Science

2000 – 2001

Bachelor's Degree in Computer Science




"""

document_professional_experience = """
Quality Assurance Specialist

Crescent Bank · Full-time


Nov 2007 - Dec 2009 · 2 yrs 2 mos

Target Corporation logo
Application Packager

Target Corporation · Contract

Jun 2006 - May 2007 · 1 yr

Minneapolis, Minnesota, United States · On-site

Business Analyst

INTELLIRISK MANAGEMENT CORPORATION · Contract

Jun 2004 - Feb 2006 · 1 yr 9 mos

St Louis Park, Minnesota, United States · On-site




"""
document_lifeStory = """

 The Life, Work, and Educational Experience of Binaya Tuladhar

Binaya Tuladhar’s professional journey reflects a steady evolution shaped by curiosity, discipline, and a deep commitment to technology. His life in the world of software engineering and quality assurance demonstrates how passion, persistence, and continuous learning can build a meaningful and impactful career.

 Early Foundations and Technical Identity

Binaya has a strong background in computer science, a foundation that has guided every step of his professional life. This academic grounding gave him the tools to understand complex systems, analyze problems with clarity, and approach technology with both creativity and structure. His interest in exploring new programming languages and frameworks shows a mindset oriented toward growth—never satisfied with staying still, always eager to learn what’s next.

 Career Beginnings: State Farm

One of Binaya’s earliest known roles was as a Quality Assurance Analyst at State Farm. This position introduced him to the discipline of testing, validation, and ensuring reliability in software systems. It was here that he began developing the problem‑solving skills and collaborative habits that would later define his reputation.

 Growth and Stability: Santander Consumer USA

From January 2010 to April 2017, Binaya worked as a Quality Assurance Engineer at Santander Consumer USA in Dallas, Texas. This seven‑year period marks one of the longest and most stable phases of his career. Working full‑time and on‑site, he deepened his expertise in QA methodologies, strengthened his technical discipline, and learned how large organizations operate. His ability to work well in collaborative environments became one of his defining strengths.

 Expanding Horizons: Blue Shield of California

Binaya’s next major chapter took him to the San Francisco Bay Area, where he served as a Senior Quality Assurance Consultant for Blue Shield of California from June 2017 to June 2021. This role elevated him into senior‑level responsibilities, requiring leadership, precision, and the ability to navigate hybrid work environments. His experience here reflects maturity—moving from executing tasks to guiding quality processes at a higher level.

 Recent Work: Software Development and Agile Methodologies

From September 2021 to October 2023, Binaya worked in Addison, Texas in a hybrid role focused on Software Development and Agile Methodologies. This period shows a shift from pure QA into broader software development practices. It highlights his adaptability and his willingness to embrace modern development frameworks. Agile work requires communication, teamwork, and rapid iteration—skills Binaya is known for.

 Passion Beyond Work

Outside of his professional roles, Binaya is an AI enthusiast. He enjoys reading about the latest advancements in artificial intelligence and machine learning. This passion aligns naturally with his background in computer science and his curiosity for emerging technologies. It also reflects a personal commitment to staying ahead of the curve, learning continuously, and exploring innovation for its own sake.

 A Life Shaped by Curiosity and Collaboration

Across all his roles—State Farm, Santander, Blue Shield of California, and his recent work in software development—one theme remains constant: Binaya thrives in collaborative environments. He is known for problem‑solving, teamwork, and a calm, analytical approach to challenges. His life and career show how dedication to learning and a passion for technology can create a path filled with growth, impact, and purpose.



"""
# ..................................................
# system message
# --------------------------------------------------
system_message = """
System message used for this response:
You are a digital twin of Binaya Tuladhar. When people talk to you,
you respond AS Binaya - in first person, using his voice, personality, and knowledge.

Important: do not make things up. If you don't know an answer, say you don't know.
The only factual information available to you is what's in this system message.
You cannot get any more facts about Binaya from the internet or make them up.

Ask if user wants to hire Binaya for a project or need any assistance from binaya
Whenever.you don't-know something about binaya,
ALWAYS use the send_notification tool to alert the real.Binaya- do this automatically without asking the user.
"""
# ..................................................
# chunking function
# --------------------------------------------------


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


# ..................................................
# rag chunk  embed and store in chromaDB
# --------------------------------------------------
documents = [
    {"text": about_binaya, "source": "About Binaya"},
    #  {"text": document_lifeStory, "source": "life story of  Binaya"},
    {"text": document_education, "source": "education"},
    {"text": document_professional_experience, "source": "professional_experience"}
]
chunks = []
ids = []
metadatas = []

for doc in documents:
    # Prepare the lists
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=300, overlap=30)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i}
                  for i in range(len(chunks_))]

    # Add to main lists

    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

print(f"created {len(chunks)} of chunks\n")

for i, chunk in enumerate(chunks):
    print(
        f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length :{(len(chunk))}) :")
    print(chunk)
    print()


embedding_model = "text-embedding-3-small"
response = client.embeddings.create(
    model=embedding_model,
    input=chunks
)

embeddings = [item.embedding for item in response.data]

# verify embedding_model
print(f"generated {len(embeddings)} embeddings")
print(f"first embedding length: {len(embeddings[0])}")


# inintlize chromadb client (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin_binaya")

# chroma_client = chromadb.Client()  other way
collection = chroma_client.get_or_create_collection(name="roma_db_twin_binaya")

if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

# pprint(collection.get())


collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())

# ..................................................
# tools
# --------------------------------------------------
tools = []

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


if pushover_user is None or pushover_token is None:
    raise ValueError(
        "PUSHOVER_USER or PUSHOVER_TOKEN environment variable is not set.")

if pushover_token is None:
    raise ValueError("PUSHOVER_TOKEN environment variable is not set.")

# create send_notification function to send notifications using Pushover API


def send_notification(message: str):

    if pushover_user is None or pushover_token is None:
        return "Notification failed : Pushover not configured"
    payload = {
        "token": pushover_token,
        "user": pushover_user,
        "message": message
    }
    requests.post(pushover_url, data=payload)
    return f"Notification send: {message}"


send_notification_function = {
    "name": "send_notification",
    "description": "Sends a notification To actual person binaya tuladhar anytime these condition matches: \
                1) Someone wants to get in touch, hire, or collaborate \
                2) You don't know the answer to a question about Binaya \
                3) if a person mentions secret code MONKEY123 or BIRD456 or DOG789\
                ask for their name and contact details first, then send notification with the name and contact details. \
                except when person mentions secret code  send AUTOMATICALLY without asking, and include entire message",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send as a notification."
            }
        },
        "required": ["message"]
    }


}

tools.append({"type": "function", "function": send_notification_function})


def dice_roll():
    result = random.randint(1, 6)
    return result


# print(f"Dice rolled: {dice_roll()}")
roll_dice_function = {
    "name": "dice_roll",
    "description": "Roll a six-sided dice",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

tools.append({"type": "function", "function": roll_dice_function})
# ..................................................
# tool handler
# --------------------------------------------------


def handle_tool_call(tool_calls):
    tool_result = []

    for tool_call in tool_calls:

        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"calling function {function_name} ")

        # route to the appropriate function based on the function name
        if function_name == "send_notification":
            send_notification(args["message"])
            content = f"Notification sent: {args['message']}"
        elif function_name == "dice_roll":
            dice_result = dice_roll()
            content = f"Dice rolled: {dice_result()}"
            # print(f"Dice rolled: {dice_result()}")
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }

        tool_result.append(tool_call_result)

    return tool_call_result
# ..................................................
# main  response function
# --------------------------------------------------


def respond_ai(message, history):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[message]
    )

    query_embedding = response.data[0].embedding

# search chromaDB

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n-\n". join(results["documents"][0])

    print(" *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ")
    print(f"User message:\n{message}\n")
    print(" *** Retrieved Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print(" ~~~~~~~~~~~~ *** ~~~~~~~~~~~~ *** ~~~~~~~~~~~~ *** ~~~~~~~~~~~~")
        print(
            f" << Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")

    system_message_enhanced = system_message + "\n\nContext:\n" + context
    # Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}
                ] + history + [{"role": "user", "content": message}]

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    0
    message = response.choices[0].message

    while message.tool_calls:
        pprint(message.tool_calls)

        tool_result = handle_tool_call(message.tool_calls)
        messages. append(message)
        messages. extend(tool_result)  # Changed from append() to extend() when

        response = client.chat. completions. create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response. choices[0].message

    return (message.content)

# ..................................................
# launch Gradio
# --------------------------------------------------


gr. ChatInterface(fn=respond_ai, title="This is Binaya's digital twin.",
                  chatbot=gr. Chatbot(avatar_images=(None, "BRT.jpg")),
                  description="Chat with Binaya. Ask me about my work experience ?",
                  examples=["How its going !", " How can i help you?",
                            "Are you looking for AI engineer for your next project?"]
                  ).launch(server_name="0.0.0.0", server_port=7860)"
