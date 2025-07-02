import os


class OpenRouterLLM:
    def __init__(self, model_id, system_prompt=None, api_key=None):

        from openai import OpenAI

        if model_id is None: raise ValueError("model_id cannot be None")
        if api_key is None: api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key: raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.model_id = model_id
        self.system_prompt = system_prompt
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self._last_prompt_messages = None

    def prompt(self, messages, temperature=0.0):
        if type(messages) is str: messages = [{"role": "user", "content": messages}]

        if messages[0]['role'] != 'system' and self.system_prompt:
             messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }   
             ] + messages
        
        self._last_prompt_messages = messages
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature
        )
        content = response.choices[0].message.content
        return content
