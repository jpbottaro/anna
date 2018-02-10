from anna.model.decoder.feedforward import FeedForwardDecoder


def test_encode_decode():
    rnn = FeedForwardDecoder(None, ["one", "two", "three"], 1, 1, False, False)
    labels = [set(["one"]), set(["two", "one"]), set(["three"]), set()]
    assert [set(l) for l in rnn.decode(rnn.encode(labels))] == labels
