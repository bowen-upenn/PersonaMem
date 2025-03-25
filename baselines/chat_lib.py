import json

from openai import OpenAI


class chatSession:
    def __init__(self, messages=None, openai_api_key=None, model="gpt-4o-mini", temperature=0):
        self.system_msg = [
            {"role": "system", "content": "You are an AI that answers only with A, B, C, or D."}
        ]
        self.msg_history = []
        self.openai_client = OpenAI(api_key=openai_api_key)

        if messages is not None:
            for idx, msg in enumerate(messages):
                self.msg_history.append(msg)
        self.model = model
        self.temperature = temperature
        self.json_schema = {
            "name": "answer_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }

    def add_message(self, role, content):
        self.msg_history.append({"role": role, "content": content})

    def update_system_msg(self, system_msg):
        self.system_msg = [{"role": "system", "content": system_msg}]

    def get_message(self, response_format="json_schema", update_history=True, **kwargs):
        input_messages = self.system_msg + self.msg_history
        input_messages = [
            {"role": message["role"], "content": str(message["content"])}
            for message in input_messages
        ]
        response = self.openai_client.chat.completions.create(
            model=self.model,
            response_format={"type": response_format, "json_schema": self.json_schema},
            messages=input_messages,
            temperature=self.temperature,
        )
        output_msg = response.choices[0].message.content
        if response_format == "json_schema":
            try:
                output_msg = json.loads(output_msg)
            except Exception as e:
                print(f"Invalid JSON format from OpenAI. Error: {e}.")
                print(output_msg)
                return self.get_message(
                    response_format="json_schema", update_history=True, **kwargs
                )
        if update_history:
            self.msg_history.append({"role": "assistant", "content": output_msg})
        return output_msg

    def clear_history(self):
        self.msg_history = []


def split_on_system(msg_history_all):
    msg_histories = []
    msg_history_curr = []
    for turn in msg_history_all:
        if turn["role"] == "system":
            msg_histories.append(msg_history_curr)
            msg_history_curr = []
        msg_history_curr.append(turn)
    msg_histories.append(msg_history_curr)
    msg_histories = msg_histories[1:]  # remove first empty list
    return msg_histories
