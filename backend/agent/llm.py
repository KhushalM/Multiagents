import os
import json
import openai
import dotenv
from typing import Optional, List, Dict

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLM:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Optional: you can keep this for documentation in the system prompt,
        # but the actual tool wiring is done via the `tools` param below.
        self.tool_call_str = """
        {
            "type": "function",
            "function": {
                "name": "search_doordash",
                "description": "Search for a food item on Doordash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the website to search"},
                        "search_term": {"type": "string", "description": "The food item to search for"},
                        "location": {"type": "string", "description": "The location to search in"}
                    },
                    "required": ["url", "search_term", "location"]
                }
            }
        }
        """

        self.system_prompt = """
        You are a helpful assistant that can use tools.
        You have access to these tools:
        {tool_list}
        Example tool call (for reference only):
        {tool_call_str}
        """

    def get_response_with_tools(self, prompt: str, tools: Optional[List[Dict]] = None):
        tools = self.tool_list()
        tool_list_text = "\n".join([f"- {t['function']['name']}: {t['function']['description']}" for t in tools])
        system_prompt = self.system_prompt.format(tool_list=tool_list_text, tool_call_str=self.tool_call_str)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
        )

        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            results = []
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                results.append(self.tool_response(name, args))
            return results[0] if len(results) == 1 else results
        return msg.content

    def tool_list(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_doordash",
                    "description": "Search for a food item on Doordash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL of the website to search"},
                            "search_term": {"type": "string", "description": "The food item to search for"},
                            "location": {"type": "string", "description": "The location to search in"},
                        },
                        "required": ["url", "search_term", "location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "parse_html",
                    "description": "Parse the HTML of a website",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "html": {"type": "string", "description": "The HTML of the website to parse"},
                        },
                        "required": ["html"],
                    },
                },
            },
        ]

    def tool_response(self, tool_name: str, tool_args: dict):
        # Simple dispatcher; no need to change your tool functions
        if tool_name == "search_doordash":
            from tools import search_doordash
            return search_doordash(**tool_args)
        if tool_name == "parse_html":
            from tools import parse_html
            return parse_html(**tool_args)
        raise ValueError(f"Unknown tool: {tool_name}")

if __name__ == "__main__":
    user_input = input("Enter a prompt: ")
    llm = LLM()
    print(llm.get_response_with_tools(user_input))