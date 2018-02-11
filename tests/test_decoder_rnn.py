from anna.model.decoder.rnn import RNNDecoder


def test_encode_decode():
    rnn = RNNDecoder(None, ["one", "two", "three"], 1)
    labels = [["one"], ["two", "one"], ["three"], []]
    assert rnn.decode(rnn.encode(labels)) == labels
