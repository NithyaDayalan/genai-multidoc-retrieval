## EXP-4 : Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM :
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT :
As the volume of research papers increases, efficiently retrieving and synthesizing information from multiple documents remains a challenge. This project aims to address the problem by creating a system that uses the LlamaIndex framework to retrieve and synthesize data from various academic papers. The system is designed to respond accurately to complex queries and provide insightful comparisons of the extracted information.

### DESIGN STEPS :
#### STEP 1 : 
Define URLs of PDFs and corresponding local filenames.

#### STEP 2 :
Define URLs of PDFs and corresponding local filenames.

#### STEP 3 :
Generate vector and summary tools for each paper.

#### STEP 4 :
Initialize the agent with all tools and the LLM.

#### STEP 5 :
Query the agent and print the responses.

### PROGRAM :
```
from helper import get_openai_api_key
from utils import get_doc_tools
from pathlib import Path
import nest_asyncio
import requests
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

nest_asyncio.apply()
OPENAI_API_KEY = get_openai_api_key()
llm = OpenAI(model="gpt-3.5-turbo")

urls = [
    "https://openreview.net/attachment?id=0OOqC20mAf&name=pdf",
    "https://openreview.net/attachment?id=3WG1ghE0xj&name=pdf",
    "https://openreview.net/attachment?id=NvUvfY9Wxi&name=pdf",
]

papers = [
    "1_Band_gap_prediction_of_prist.pdf",
    "2_Towards_a_Database_of_Bond_O.pdf",
    "3_Regression_Models_for_Lattic.pdf",
]

for url, paper in zip(urls, papers):
    r = requests.get(url)
    with open(paper, "wb") as f:
        f.write(r.content)

paper_to_tools = {}
for paper in papers:
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools[paper]]

agent_worker = FunctionCallingAgentWorker.from_tools(
    all_tools,
    llm=llm,
    verbose=True
)
agent = AgentRunner(agent_worker)

response_1 = agent.query(
    "What are the key features or descriptors used as input to ML models in materials science?, "
    "and How do you validate and test the accuracy of ML models for material property prediction?"
)
print(str(response_1))

response_2 = agent.query(
    "Give me a summary of benefits of using ML over traditional computational methods"
)
print(str(response_2))
```

### OUTPUT :
<img width="1230" height="708" alt="image" src="https://github.com/user-attachments/assets/c748382f-597e-497f-8b79-79a57e462f77" />
<img width="969" height="763" alt="image" src="https://github.com/user-attachments/assets/ef89c361-a8d7-4fdf-8605-9b82d6830c59" />

### RESULT :
The multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The agent demonstrated its capability to retrieve and synthesize information from multiple academic papers, answering complex queries with concise, relevant, and accurate responses.
