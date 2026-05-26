answers = {
    1: 'D', # Input A, B, C, D, or E.
    2: 'D', # Input A, B, C, D, or E.
    3: 'D', # Input A, B, C, D, or E.
    4: 'D', # Input A, B, C, D, or E.
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
