

def shim_decorator(func):
    def shim(*args, **kwargs):
        print(f"Function '{func.__name__}' called with arguments: {args} and keyword arguments: {kwargs}.")
        return func(*args, **kwargs)
        print("After calling the func.")
    return shim


@shim_decorator
def greeting(name, greeting="Hello"):
    return f"{greeting}, {name}!"


result = greeting("Theo", greeting="Hi")
print(result)
