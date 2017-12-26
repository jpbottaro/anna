from anna.nlp.utils import tokenize

def test_simple():
    assert tokenize("This is text") == \
        ["This", "is", "text"]

def test_remove():
    assert tokenize("This^ is text *") == \
        ["This", "is", "text"]

def test_separate():
    assert tokenize("I'm a test, super-test.") == \
        ["I", "'m", "a", "test", ",", "super-test", "."]

def test_number():
    assert tokenize("The numbers 1, 2 and 1,243.32 exist") == \
        ["The", "numbers", "1", "1", "and", "1", "exist"]
