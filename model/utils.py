import os


def svuota_cartella(cartella):
    # Controlla se la cartella esiste
    if os.path.exists(cartella):
        # Elenca tutti i file nella cartella e li rimuove
        for file in os.listdir(cartella):
            file_path = os.path.join(cartella, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Errore durante la rimozione di {file_path}: {e}")
