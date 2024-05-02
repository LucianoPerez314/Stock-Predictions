"""
Stock market prediction using Markov chains.

For each function, replace the return statement with your code.  Add
whatever helper functions you deem necessary.
"""

import comp140_module3 as stocks
import random

### Model
from collections import defaultdict
   


def transition_p(bins, order):
    """
    Calculates the transition probabilities of the price changing to another bin from the given data
    with the given order.
    
    inputs:
        - bins: a list of integers representing bins
        - order: an integer repesenting the desired order of the markov chain
    
    return:
        - transition_probabilities: a dictionary
    """
    trans_p = {}
    sample1 = defaultdict(int)
    sample2 = defaultdict(int)
    
    full_sequences = []
    partial_sequences = []
    if order == 0:
        for binz in bins:
            trans_p[binz] += 1/len(bins)
    #initializes probabilities
    else:   
        for idx in range(len(bins)):
            tup = tuple(bins[idx: idx + order + 1]) 
            if len(tup) == order + 1:
                full_sequences.append(tup)
    
    #creates a list of all the full sequences with n order (INCLUDES REPEATS)
        for tup in full_sequences:       
            partial = tup[0: order]
            partial_sequences.append(partial)
     
    # creates a list of all the partial sequences with n order (INCLUDES REPEATS)
        for tup in full_sequences:
            sample1[tup] += 1     

    #creates a dictionary of how many times a full sequence tuple repeats
        for par in partial_sequences:
            sample2[par] += 1
    #creates a dictionary of how many times a partial sequence tuple repeat
        for key2 in sample2.keys():
            sub_dict = {}
            trans_p[key2] = sub_dict
            for key1 in sample1.keys():
                if key1[0:order] == key2:
                    probability = sample1[key1] / sample2[key2]
                    sub_dict.update({key1[-1]: probability})

    return trans_p


       
def markov_chain(data, order):
    """
    Create a Markov chain with the given order from the given data.

    inputs:
        - data: a list of ints or floats representing previously collected data
        - order: an integer repesenting the desired order of the markov chain

    returns: a dictionary that represents the Markov chain
    """
    
    #converts price changes into a list of bins
    markov = transition_p(data, order)
    #organizes bin list into a markov_chain
    return markov    



def weighted_choice(choices):
    """
    Performs the random.choice() function except the weights can be modified.
    
    Input:
    choices: a list where every element is also a list or tuple containing the choice
    and the weight of that choice.
    
    Return:
    a choice
    """
    total = sum(weight for choic, weight in choices)
    rand = random.uniform(0, total)
    upto = 0
    for choic, weight in choices:
        if upto + weight >= rand:
            return choic
        upto += weight
    

### Predict

def predict(model, last, num):
    """
    Predict the next num values given the model and the last values.

    inputs:
        - model: a dictionary representing a Markov chain
        - last: a list (with length of the order of the Markov chain)
                representing the previous states
        - num: an integer representing the number of desired future states

    returns: a list of integers that are the next num states
    """
    last = last.copy()
    next_states = []
    for idx in range(num):
    #Repeats process for a given number of times
        if tuple(last) in model:
            #if the state is in markov chain
            transitions = model[tuple(last)]
            #Accesses the dictionary of given probabilities corresponding to the previous events.
        

            choices = tuple(transitions.items())
                
            
            next_state = weighted_choice(choices)
            next_states.append(next_state)
            last.remove(last[0])
            last.append(next_state)
            
        else:
            next_state = random.randrange(0,4)
            next_states.append(next_state)
            last.remove(last[0])
            last.append(next_state)
            
    #Makes a new markov chain and makes it the model and makes new history the last
    #Error because 
    return next_states

    
        



### Error

def mse(result, expected):
    """
    Calculate the mean squared error between two data sets.

    The length of the inputs, result and expected, must be the same.

    inputs:
        - result: a list of integers or floats representing the actual output
        - expected: a list of integers or floats representing the predicted output

    returns: a float that is the mean squared error between the two data sets
    """
    summate = 0
    for idx in range(len(result)):
        diff = (expected[idx] - result[idx]) ** 2
        summate += diff
    mean_squared_error = summate/len(result)
    return mean_squared_error


### Experiment

def run_experiment(train, order, test, future, actual, trials):
    """
    Run an experiment to predict the future of the test
    data given the training data.

    inputs:
        - train: a list of integers representing past stock price data
        - order: an integer representing the order of the markov chain
                 that will be used
        - test: a list of integers of length "order" representing past
                stock price data (different time period than "train")
        - future: an integer representing the number of future days to
                  predict
        - actual: a list representing the actual results for the next
                  "future" days
        - trials: an integer representing the number of trials to run

    returns: a float that is the mean squared error over the number of trials
    """
    mses = 0
    markov = markov_chain(train, order)
    for idx in range(trials):
        predicted = predict(markov, test, future)
        experiment = mse(predicted, actual)
        mses += experiment
        avg = mses/trials
    return avg


### Application

def run():
    """
    Run application.

    You do not need to modify any code in this function.  You should
    feel free to look it over and understand it, though.
    """
    # Get the supported stock symbols
    symbols = stocks.get_supported_symbols()

    # Get stock data and process it

    # Training data
    changes = {}
    bins = {}
    for symbol in symbols:
        prices = stocks.get_historical_prices(symbol)
        changes[symbol] = stocks.compute_daily_change(prices)
        bins[symbol] = stocks.bin_daily_changes(changes[symbol])

    # Test data
    testchanges = {}
    testbins = {}
    for symbol in symbols:
        testprices = stocks.get_test_prices(symbol)
        testchanges[symbol] = stocks.compute_daily_change(testprices)
        testbins[symbol] = stocks.bin_daily_changes(testchanges[symbol])

    # Display data
    #   Comment these 2 lines out if you don't want to see the plots
    stocks.plot_daily_change(changes)
    stocks.plot_bin_histogram(bins)

    # Run experiments
    orders = [1, 3, 5, 7, 9]
    ntrials = 500
    days = 5

    for symbol in symbols:
        print(symbol)
        print("====")
        print("Actual:", testbins[symbol][-days:])
        for order in orders:
            error = run_experiment(bins[symbol], order,
                                   testbins[symbol][-order-days:-days], days,
                                   testbins[symbol][-days:], ntrials)
            print("Order", order, ":", error)
        print()

# You might want to comment out the call to run while you are
# developing your code.  Uncomment it when you are ready to run your
# code on the provided data.

run()
