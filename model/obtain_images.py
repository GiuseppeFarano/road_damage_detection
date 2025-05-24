import os
import requests
from model.utils import svuota_cartella

# Chiave API di Google
API_KEY = '' # Your Google Cloud Platform API key here
OUTPUT_IMAGE_FOLDER = 'images/output/'


# Funzione per ottenere le coordinate con Google Geocoding API
def ottieni_coordinate_geocoding_google(luogo):
    geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': luogo,
        'key': API_KEY
    }
    response = requests.get(geocoding_url, params=params)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            location = results[0]['geometry']['location']
            return location['lat'], location['lng']  # latitudine e longitudine
    return None


# Funzione per ottenere il percorso da OpenStreetMap tramite OSRM
def ottieni_percorso_osrm(partenza_coord, destinazione_coord):
    osrm_url = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}".format(
        partenza_coord[1], partenza_coord[0], destinazione_coord[1], destinazione_coord[0]
    )

    params = {
        'overview': 'full',  # Può essere 'full' o 'simplified'
        'geometries': 'geojson'  # Formato del percorso
    }

    response = requests.get(osrm_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('routes'):
            # Estrarre le coordinate del percorso (lista di [lon, lat])
            percorso = data['routes'][0]['geometry']['coordinates']
            # Convertire da [lon, lat] a [lat, lon]
            return [[coord[1], coord[0]] for coord in percorso]
    return None


# Funzione per estrarre punti intermedi lungo il percorso
def estrai_punti_intermedi(percorso, num_punti=10):
    # Calcola la distanza tra i punti selezionati
    if num_punti >= len(percorso):
        return percorso
    step = len(percorso) // num_punti
    punti_intermedi = [percorso[i] for i in range(0, len(percorso), step)]
    return punti_intermedi[:num_punti]


# Funzione per scaricare immagini da Google Street View
def scarica_immagini(coordinate):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    # Verifica se la cartella 'images' esiste, altrimenti la crea
    if not os.path.exists('images/input'):
        os.makedirs('images/input')
    else:
        svuota_cartella("images/input")

    for coord in coordinate:
        params = {
            'size': '600x600',  # Dimensioni dell'immagine
            'location': f"{coord[0]},{coord[1]}",  # latitudine, longitudine
            'fov': 90,          # Campo visivo
            'heading': 90,     # Direzione della visuale
            'pitch': -10,         # Angolazione verticale
            'key': API_KEY      # API key di Google Cloud Platform
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            # Salva l'immagine nella cartella 'images'
            # Usa la funzione per svuotare la cartella prima di salvare l'immagine
            with open(f"images/input/street_view_{coord[0]:.6f}_{coord[1]:.6f}.jpg", 'wb') as f:
                f.write(response.content)
