# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
def add(a, b): 
    """  This function shuold return the sum of a and b. I.e. if print(add(2,3)) should print '5'. """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def misterfy(animals : list[str]) -> list[str]: 
    """
    Given a list of animals like animals=["cat", "wolf", "elephans"], this function should return
    a list like ["mr cat", "mr wolf", "mr elephant"]  """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def mean_value(p_dict : dict) -> float: 
    """
    Given a dictionary of the form: {x: probability_of_x, ...} compute the mean value of
    x, i.e. sum_i x_i * p(x_i). The recommended way is to use list comprehension and not numpy.
    Hint: Look at the .items() method and the build-in sum(my_list) method. """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")


class BasicFruitShop:
    """This is a simple class that represents a fruit-shop.
    You should give it a dictionary of prices when you initialize it."""
    def __init__(self, name : str, prices : dict):
        """prices is a dictionary of the form {fruit_name: cost}. For instance
        prices = {'apple': 5, 'orange': 6} """
        self.name = name
        self.prices = prices

    def cost(self, fruit : str) -> float: 
        """ Return the cost in pounds of the fruit with name 'fruit'. It uses the self.prices variable
        to get the price.
        You don't need to do exception handling here. """
        # TODO: 1 lines missing.
        raise NotImplementedError("Return cost of fruit as a floating point number")


if __name__ == '__main__':
    "This code runs when you invoke the script from the command line (but not otherwise)"

    """ Quesion 1: Lists and basic data types """
    print("add(2,5) function should return 7, and it returned", add(2, 5))  

    animals = ["cat", "giraffe", "wolf"] 
    print("The nice animals are", misterfy(animals)) 

    """  
    This problem represents the probabilities of a loaded die as a dictionary such that     
    > p(roll=3) = p_dict[3] = 0.15.
    """
    p_die = {1: 0.20,
             2: 0.10,
             3: 0.15,
             4: 0.05,
             5: 0.10,
             6: 0.40}
    print("Mean roll of die, sum_{i=1}^6 i * p(i) =", mean_value(p_die)) 

    """ Part B: A simple class """
    price1 = {"apple": 4, "pear": 8, 'orange': 10} 
    shop1 = BasicFruitShop("Alis Funky Fruits", price1)

    price2 = {'banana': 9, "apple": 5, "pear": 7, 'orange': 11}
    shop2 = BasicFruitShop("Hansen Fruit Emporium", price2)

    fruit = "apple"
    print("The cost of", fruit, "in", shop1.name, "is", shop1.cost(fruit))
    print("The cost of", fruit, "in", shop2.name, "is", shop2.cost(fruit)) 
