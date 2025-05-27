import pickle
with open('pitchnames.pkl', 'rb') as f:
    pitchnames = pickle.load(f)
print(len(pitchnames))
