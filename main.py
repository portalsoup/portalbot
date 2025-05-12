import argparse

from src.AIPipeline import AIPipeline, Response
from src.AIPipelineContext import AIPipelineContext
from src.globals import init_cuda, load_prompt, init_args, handle_if_quit

class App:
    ctx: AIPipelineContext
    ai: AIPipeline
    messages: list[dict[str, str]]

    def __init__(self, args: argparse.Namespace):
        if not args.no_cuda:
            init_cuda()

        self.ctx = AIPipelineContext(args.model, verbose=args.verbose, think=args.think)
        self.ai = AIPipeline(self.ctx)
        self.messages = load_prompt(args.prompt)

    def main(self):
        while True:
            self._get_user_input("Prompt: ")
            response = self._generate_response()
            self._print_response(response)

    def _append_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def _print_response(self, response: Response):
        if self.ctx.think:
            print(f"${response.thinking.strip()}\n\nResponse:\n")

        print(response.response.strip())

    def _get_user_input(self, prompt: str):
        inputted = input(prompt)
        handle_if_quit(inputted)
        self._append_message("user", inputted)
        return inputted

    def _generate_response(self) -> Response:
        response = self.ai.query(self.messages)
        self._append_message("assistant", response.response)
        return response

if __name__ == "__main__":
    app = App(init_args())  # Create an instance of the class
    app.main()  # Call the main method