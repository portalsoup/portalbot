from src.AIPipeline import Message, AIContext, AIPipeline2
import torch

if torch.cuda.is_available():
    print("GPU available")
    torch.cuda.empty_cache()

ctx = AIContext(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "auto",
    "text-generation",
    2048, # 32768,
    torch.float16
)

ai = AIPipeline2(ctx)
ai.add_template(Message(content="You respond in english."))
ai.add_template(Message(content=" Your name is Mr. Butlertron the robot butler."))
r = ai.query(Message("What is your name?"))
# r = ai.query(Message("Give me a short introduction to large language model."))
print(r)
quit(0)
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in {"exit", "quit"}:
#         print("Exiting...")
#         break
#
#     message = Message(user_input)
#     response = ai.query(message)
#     print(response)
