import numpy as np

def to_seq_length(seq_list, SL, freq):

    seqs = []
    for orig_seq in [x for x in seq_list if x.shape[1] >= SL*freq]:

        for sampled_s in [orig_seq[:, i::freq] for i in range(freq)]:

            seqs.extend( [sampled_s[:, i:i+SL] for i in range(sampled_s.shape[1]-SL)] ) 

    return np.array(seqs)


def make_dataset(SL=30, freq=2):
        
    train = np.load("./data/handball_shot/ri_hbs_train.npy")

    train = to_seq_length(train, SL, freq)
    np.save("./data/handball_shot/train", train)
    np.save("./data/handball_shot/valid", train[::SL])
    del train

    test = np.load("./data/handball_shot/ri_hbs_test.npy")
    test = to_seq_length(test, SL, freq)
    np.save("./data/handball_shot/test", test[::SL])

    
make_dataset()
print("done!")