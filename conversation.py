from llama_cpp import Llama
from globals import SYSTEM_MSG


class Chat:
    def __init__(self, model_path, user_name, self_name):
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=False,
        )
        self.user_name = user_name
        self.self_name = self_name
        self.memory = None
        self.sys_message = SYSTEM_MSG

    def prepare_message(self, message):
        tokens = self.assemble_tokents(message)
        generator = self.model.generate(
            tokens,
            top_k=30,
            top_p=0.90,
            temp=0.2,
            repeat_penalty=1.1,
        )
        output_message = []
        for i, token in enumerate(generator):
            token_str = self.model.detokenize([token]).decode("utf-8", errors="ignore")
            tokens.append(token)
            if token == self.model.token_eos() or len(tokens) > 2000 - 1:
                break
            yield token_str
        if self.memory:
            self.memory.process("".join(output_message), self.self_name)

    def assemble_tokents(self, message):
        system_tokens = self.model.tokenize(
            f"<s>system\n{self.sys_message}\n</s>".encode("utf-8"), special=True
        )
        role_tokens = self.model.tokenize(
            f"{self.self_name}:\n".encode("utf-8"), special=True
        )
        message_tokens = self.model.tokenize(
            f"<s>{self.user_name}\n{message}\n</s>".encode("utf-8"), special=True
        )
        tokens = system_tokens + role_tokens
        if self.memory:
            tokens += self.memory.process(message, self.user_name).encode("utf-8")
        else:
            tokens += message_tokens
        return tokens


class Memory:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.messages = []

    def process(self, message: str, actor_name: str) -> str:
        if len(self.messages) >= self.buffer_size:
            self.messages.pop(0)

        self.messages.append((actor_name, message))

        return "\n".join([f"{name}: {msg}" for name, msg in self.messages])
