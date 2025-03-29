import sys


def get_size_repartition(obj, details=False) -> dict[str, int|dict[str, int]]:
    res = {}
    for attribute in dir(obj):
        if not attribute.startswith('_'):
            value = getattr(obj, attribute)
            if details and hasattr(value, '__dict__'):
                res[attribute] = get_size_repartition(value)
            else:
                res[attribute] = sys.getsizeof(value)
    return res


def print_size_repartition(obj, details=False):
    memory = get_size_repartition(obj, details)
    for key, value in memory.items():
        if type(value) == int:
            print(f'{key} : {value:,}')
        else:
            print(f'{key} : {sum(value.values()):,}')
            for key_, value_ in value.items():
                print(f'- {key_} : {value_:,}')
