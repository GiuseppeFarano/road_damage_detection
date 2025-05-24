# Importazione delle librerie necessarie
from flask import Flask, render_template  # Flask per la creazione dell'app web e render_template per il rendering delle pagine HTML
from controllers.obtain_images import obtain_images_bp  # Importazione del Blueprint per la gestione delle immagini
from controllers.inference import inference_bp  # Importazione del Blueprint per la gestione dell'inferenza

# Creazione dell'istanza dell'app Flask
app = Flask(__name__)

# Impostazione di una chiave segreta per gestire le sessioni e la sicurezza (ad es. CSRF)
app.secret_key = '' # Your secret key here

# Registrazione dei Blueprint per modularizzare le funzionalità dell'app
app.register_blueprint(obtain_images_bp)  # Blueprint per le operazioni sulle immagini
app.register_blueprint(inference_bp)  # Blueprint per le operazioni di inferenza


# Definizione della route principale per il rendering della pagina HTML
@app.route('/', methods=['GET'])  # Associa l'URL '/' al metodo index e accetta solo richieste GET
def index():
    """
    Route principale che renderizza la pagina index.html.
    Il parametro `right_panel_flag` è passato al template per determinare
    se mostrare o meno il pannello destro sulla pagina.
    """
    return render_template('index.html', right_panel_flag=True)


# Esecuzione dell'applicazione Flask
if __name__ == '__main__':
    # Avvia il server Flask in modalità debug per un rapido sviluppo
    app.run(debug=True)
