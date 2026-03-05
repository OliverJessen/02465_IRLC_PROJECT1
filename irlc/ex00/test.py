class BetterBasicDog():
    def __init__(self, name):
        self.name = name
        self.age = 0


    def birthday(self):
        self.age += 1
        print(f"Happy birthday {self.name}! You are now {self.age} years old.")


d1 = BetterBasicDog("Rover") # Create an instance of BetterBasicDog with the name "Rover"
d1.birthday() # Call the birthday method to increment the age and print the message (should be 1)
d1.age = 5 # Manually set the age to 5
d1.birthday() # Call the birthday method again to increment the age and print the message (should be 6)