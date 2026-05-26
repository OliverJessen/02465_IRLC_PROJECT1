answers = {
    1: 'E', # Input A, B, C, D, or E.
    2: 'E', # Input A, B, C, D, or E.
    3: 'E', # Input A, B, C, D, or E.
    4: 'E', # Input A, B, C, D, or E.
    5: 'E', # Input A, B, C, D, or E.
    6: 'E', # Input A, B, C, D, or E.
    7: 'E', # Input A, B, C, D, or E.
    8: 'E', # Input A, B, C, D, or E.
    9: 'E', # Input A, B, C, D, or E.
}

def check_answers():
    for k, v in answers.items():
        if v is None:
            pass # This is ok.
        elif not isinstance(v, str):
            raise Exception(f"> Bad answer '{v}' to question {k}. Your answer must be a string.")
        elif not v.upper() in 'ABCDE':
            raise Exception(f"> Bad answer '{v}' to question {k}. Your answer must be a single letter A, B, C, D or E.")


if __name__ == "__main__":
    for k, v in answers.items():
        print(f"Question {k}: you answered '{v}'")
    check_answers()
