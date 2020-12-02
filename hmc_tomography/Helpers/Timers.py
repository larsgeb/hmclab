""" A class that allows one to time functions over many repeated calls. Can be applied
to instance methods.

Usage:

```
def my_function(a):
    return a * 100.0

my_function(5.0) # Does not log total execution time.

my_function = AccumulatingTimer(my_function)

my_function(5.0) # Now logs total execution time.

my_function.print_statistics() # Look into the statistics

```

"""
from datetime import datetime as _datetime


class AccumulatingTimer:
    def __init__(self, function):
        """Constructor that initializes number of calls and time spent."""

        if type(function) == AccumulatingTimer:
            # If we try to apply AccumulatingTimer twice, extract function from the
            # encapsulating object
            self.function = function.function
        else:
            self.function = function

        self.time_spent = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        """Call the function, keeping track of time and number of calls."""

        self.calls += 1
        start = _datetime.now()
        return_value = self.function(*args, **kwargs)
        end = _datetime.now()
        self.time_spent += (end - start).total_seconds()

        # Return original value
        return return_value

    def print_statistics(self):
        print(f"Function: {self.function}")
        print(f"Calls: {self.calls}")
        print(f"Total time spent: {self.time_spent} seconds")

    def __str__(self) -> str:
        return (
            f"Function: {self.function}, "
            f"Calls: {self.calls}, "
            f"Total time spent: {self.time_spent} seconds, "
        )
