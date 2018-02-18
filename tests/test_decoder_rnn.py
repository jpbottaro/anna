from anna.model.decoder.rnn import RNNDecoder


def test_encode_decode():
    rnn = RNNDecoder(None, ["one", "two", "three"], 1)
    labels = [["one"], ["two", "one"], ["three"], []]
    to_set = lambda x: [set(l) for l in x]
    assert to_set(rnn.decode(rnn.encode(labels))) == to_set(labels)
