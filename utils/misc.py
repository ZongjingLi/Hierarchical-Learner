

def num2word(i):
    assert i < 10
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return numbers[i]
    
def copy_dict(d):
    return d if not isinstance(d, dict) else {k: copy_dict(v) for k, v in d.items()}
