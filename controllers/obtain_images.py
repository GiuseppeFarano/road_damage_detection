from flask import Blueprint, request, session
from model.obtain_images import ottieni_coordinate_geocoding_google, ottieni_percorso_osrm, estrai_punti_intermedi, scarica_immagini
from flask import render_template

obtain_images_bp = Blueprint('obtain_images', __name__)


def ottieni_coordinate(partenza, destinazione):
    partenza_coord = ottieni_coordinate_geocoding_google(partenza)
    destinazione_coord = ottieni_coordinate_geocoding_google(destinazione)

    # Calcola il percorso e ottieni le coordinate di tutti i punti che formano il percorso con OSM
    percorso = ottieni_percorso_osrm(partenza_coord, destinazione_coord)

    # Estrai un certo numero di punti intermedi, numero che al momento viene impostato direttamente nella funzione
    punti_intermedi = estrai_punti_intermedi(percorso)
    return punti_intermedi


@obtain_images_bp.route('/ottieni_immagini', methods=['GET'])
def ottieni_immagini():
    partenza = request.args.get('partenza')
    destinazione = request.args.get('destinazione')
    flag = False

    try:
        # Ottieni con Google Maps le coordinate del punto di partenza e di arrivo a partire dalla località in formato testuale
        punti_intermedi = ottieni_coordinate(partenza, destinazione)
        # Sfrutta le coordinate dei punti estratti per ottenere le immagini da Street View
        scarica_immagini(punti_intermedi)

        text = 'Coordinate e immagini salvate con successo!'
        flag = True

    except Exception as e:
        text = 'Errore nel download delle immagini! Selezionare l\'indirizzo di partenza e destinazione fra uno di quelli suggeriti'

    # Salviamo nella sessione (per evitare di passarli tramite URL) la località di partenza e arrivo, ci servirà in futuro
    parti_partenza = partenza.split(', ')
    parti_partenza = parti_partenza[:-2]
    partenza = ', '.join(parti_partenza)
    parti_destinazione = destinazione.split(', ')
    parti_destinazione = parti_destinazione[:-2]
    destinazione = ', '.join(parti_destinazione)
    session['partenza'] = partenza
    session['destinazione'] = destinazione
    return render_template('obtain_images.html', text=text, flag=flag, right_panel_flag=True)
