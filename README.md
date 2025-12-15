# Adaptive Tool Agent (Adaptive Tooling Agent)

This project implements a robust **AI Agent** built with LangChain and a local Large Language Model (Mistral). The agent intelligently switches between different processing paths—calculation, real-time search, and knowledge-based response—to provide accurate and timely answers.

## Overview

The agent analyzes the user's query and automatically selects the optimal processing route based on a defined priority system. This architecture effectively mitigates the common weaknesses of conventional LLMs, such as inaccurate calculations and the inability to access up-to-date information, by leveraging external tools.

### Key Features

* Priority-Based Routing (Guardrails): Critical queries like currency conversion and **natural language arithmetic** are routed with the highest priority, preventing misinterpretation by the LLM.
* Calculation Accuracy: Complex arithmetic and **ambiguous natural language math** is offloaded from the LLM to a dedicated 'calculate' tool utilizing **rule-based expression generation**, ensuring mathematically reliable results.
* Real-Time Information Retrieval (RAG): The agent integrates with Google Search (via SerpAPI) to fetch current information (e.g., latest exchange rates, recent political facts) for generation.
* Robust Tool Calling: Includes a dedicated parsing logic to forcibly extract and execute tool calls even when the LLM's response is in a non-standard or malformed JSON format within the text.

### 🛡️ Core LLM Weaknesses Overcome

This agent achieves high robustness by systematically addressing three major limitations commonly found in Large Language Models (LLMs) through prioritized routing and external tool usage.

#### 1. Elimination of Calculation Errors (Hallucination)
* **The Problem**: LLMs frequently make mistakes in arithmetic, especially with complex order of operations or simple calculations phrased in natural language.
* **The Solution**: We implement a **Rule-Based Calculation Guardrail** (Priority 0.5). When a math query (either symbol-based or **natural language** like "10 apples plus 5 oranges") is detected, the agent uses a **rule-based expression generator** to determine the formula, completely bypassing the LLM's reasoning and forcing execution of the dedicated **calculate tool**, ensuring mathematical accuracy.

#### 2. Guaranteed Real-Time Information Retrieval (RAG)
* **The Problem**: An LLM's knowledge is static, making it unable to answer questions requiring the latest facts (e.g., current political leaders, recent events).
* **The Solution**: We utilize **Forced RAG Routing**. Queries seeking current information (e.g., "Who is the Prime Minister of Japan?") are automatically routed to the **`Google Search` tool**, guaranteeing a reference to the latest web data for response generation.

#### 3. Automated Complex Knowledge and Calculation Workflows
* **The Problem**: It is difficult for LLMs to reliably extract a specific numerical value from a search result (e.g., an exchange rate) and then accurately apply it in a separate calculation.
* **The Solution**: We employ a **Highly Robust Multi-Step RAG Chain**. This process coordinates: Search $\rightarrow$ **Agent Rule-based Extraction** (Numerics are forcibly normalized and extracted, eliminating LLM hallucination of rates like '15564') $\rightarrow$ Calculation tool executes the final math (e.g., "20 * [rate]"), with the final result **rounded for display accuracy** to prevent float errors.

## 🛠️ Technology Stack

| Technology | Role | License / Notes |
| :--- | :--- | :--- |
| Development Language | Python 3.x | - |
| LLM Framework | LangChain | MIT License |
| Base Model (LLM) | Mistral ('mistral:instruct') | Apache 2.0 or Mistral AI License. |
| LLM Execution | Ollama | MIT License. Used for local model management and serving. |
| Search API | SerpAPI (Google Search) | Subject to SerpAPI Terms of Service. Used for real-time web search. |

## 🚀 Setup and Environment

### 1. Install Dependencies

# Install required Python libraries
pip install langchain langchain-core serpapi

### 2. Ollama and Model Setup

1. Install the Ollama application.
2. Run the following command in your terminal to download the Mistral model:
ollama run mistral:instruct

### 3. API Key Configuration

This agent uses SerpAPI for Google Search functionality. Please set your SerpAPI key as an environment variable:

export SERPAPI_API_KEY="[YOUR_SERPAPI_KEY]"

## ⚙️ Usage (使い方)

Once the setup is complete, you can run the agent directly from your terminal.

### 1. Run the Agent

Execute the Python script to start the interactive chat session:

python agent_main.py

### 2. Example Interactions

The agent will prompt you with あなた:. You can test the different routing mechanisms using the following query types:

| Query Type | Example Query | Expected Agent Action |
| :--- | :--- | :--- |
| Calculation | 150 plus 25 times 4 | Executes the calculate tool for precise math. |
| Forced Search | Who is the current Prime Minister of Japan? | Executes the google_search tool (RAG) to find the latest fact. |
| Guarded Search | How many Japanese Yen is 100 US Dollars right now? | Triggers the high-priority RAG and calculation process. |
| Knowledge/LLM | Explain the concept of tool use in LLM agents. | Answers directly using the base LLM's internal knowledge. |

### 3. Exiting the Session

To stop the agent and end the session, type exit or quit.

あなた: exit

---

## 📜 Licensing and Copyright

### Copyright Notice

Copyright (c) 2025 Akira Hirohashi
All rights reserved.

### License

The source code for this project is released under the MIT License. Please refer to the [LICENSE](https://github.com/com-hiro/ai-agent/blob/main/LICENSE) file in this repository for the full terms and conditions.

### Acknowledgements and Credit

We extend our sincere thanks to the following open-source technologies and services used in this project:

* LangChain: For providing the framework for building reliable agents and tools.
* Ollama: For efficient local execution and management of the large language model.
* Mistral LLM (mistral:instruct): For serving as the powerful foundation model that drives the agent's intelligence.
* SerpAPI: For providing accurate, real-time web search data.

---

### 🛡️ Disclaimer

This software is provided "AS IS," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement.

The developer and copyright holders shall not, in any event, be liable for any claim, damages, or other liability arising from the use or inability to use the software. Use the program at your own risk.