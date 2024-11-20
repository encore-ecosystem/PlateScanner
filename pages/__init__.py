from .predict import view as view_predict
from .validate import view as view_validate

def view():
    print("1) Predict")
    print("2) Validate")
    print("0) Exit")
    menu = input(">> ")

    match menu:
        case '0':
            return
        case '1':
            view_predict()
        case '2':
            view_validate()
        case _:
            print("Please, choose any valid option")

__all__ = [
    'view'
]
