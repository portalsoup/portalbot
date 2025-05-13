from src.App import App
from src.globals import init_args

if __name__ == "__main__":
    app = App(init_args())  # Create an instance of the class
    app.main()  # Call the main method