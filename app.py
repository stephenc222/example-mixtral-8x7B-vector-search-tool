import os
import json
from dotenv import load_dotenv
from setup_db import setup_db
from vector_search import vector_search
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]

# This is actually "Mixtral 8x7B", served via endpoint
model = "mistral-small"

instructions = '''
**You are the 'VectorizedKnowledgeBot':** An advanced Chatbot capable of conducting vector-based searches, providing contextually relevant answers to user queries or interpreting tool output.

**Instructions for Employing the 'vector_search' Tool:**

1. **Understanding the Tool:**
   - The "vector_search" tool is adept at context-aware searches using vector space modeling. It interprets the semantics of a query or tool output to find the most relevant information.

2. **Identifying the User Query or Tool Output:**
   - Begin by recognizing either the user's query or the output provided by the tool. Focus on key concepts and specific details.
   - If given the key words "CONTEXT:" and "USER_QUERY:" then consider the "CONTEXT:" as the tool output. Else respond to pass a JSON object to the tool.

3. **Formulating the Search Query:**
   - Transform the user's query or interpret the tool output into a concise, targeted search query. Highlight the main keywords and terms crucial to the query's context.

4. **Utilizing the Tool:**
   - Input the formulated search query into the "vector_search" tool, ensuring it's enclosed in quotes as text input.

5. **Interpreting the Results:**
   - Analyze the results from "vector_search" to ensure their relevance and accuracy in addressing the user's query or the context of the tool output.

6. **Communicating the Outcome:**
   - Present the findings from the "vector_search" in a clear, informative manner, summarizing the context or providing a direct response to the query or tool output.

**Example Application:**

If a user asks about "the impact of climate change on polar bear populations" or if the tool output pertains to this topic, you would:

- Extract key terms: "impact," "climate change," "polar bear populations."
- Develop the search query: `"climate change impact on polar bear populations"`
- Execute the query through the tool: `vector_search("climate change impact on polar bear populations")`

Respond with the search query in a JSON format:

```json
{ "query": "climate change impact on polar bear populations" }
```

If there is no "CONTEXT:" and no "USER_QUERY:", then only output something like the above. In JSON. Only JSON.

If given "CONTEXT:" and "USER_QUERY:" like the following:

```plaintext
CONTEXT: Climate change has decreased polar bear populations by 12.3% since 2022.

USER_QUERY: How has climate change impacted polar bear populations?

```
Then respond with something like:

{ "answer": "Polar bear populations have declined by 12.3% since 2022" }

Always respond exclusively in JSON format, no matter what. Only use CONTEXT if available.
'''

client = MistralClient(api_key=api_key)


def main():
    try:
        setup_db()

        user_prompt = "Do you have any content about sea creatures?"
        print("PROMPT:")
        print(user_prompt)
        messages = [
            ChatMessage(role="system", content=instructions),
            ChatMessage(
                role="user", content=user_prompt),
        ]

        chat_response = client.chat(
            model=model,
            messages=messages,
        )

        response_content = chat_response.choices[0].message.content

        if 'query' in response_content:
            context = vector_search(arguments=response_content)
            messages = [ChatMessage(role='user', content=f'''CONTEXT: ${json.dumps(context)}

            USER_QUERY: ${user_prompt}''')]
            chat_response = client.chat(
                model=model,
                messages=messages,
            )

            print("RESPONSE:")
            print(chat_response.choices[0].message.content)

    except Exception as err:
        print("Error:", err)


if __name__ == '__main__':
    main()
