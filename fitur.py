fitur_model = [
    "explicit", "danceability", "energy", "mode", "speechiness", 
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]

import joblib
joblib.dump(fitur_model, 'fitur_spotify.pkl')
