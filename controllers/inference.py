from flask import Blueprint, render_template, send_from_directory, session
from model.inference import detect, generate_report, generate_score, sort_score, get_images_data

inference_bp = Blueprint('inference', __name__)
OUTPUT_IMAGE_FOLDER = 'images/output/'  # Cartella in cui sono salvate le immagini analizzate dal modello


def genera_report():
    # Genera report in cui si contano quanti danni ci sono per ogni tipologia considerando tutte le immagini
    total_damages = generate_report(file_path='images/output/detected_damages')

    # Genera file "damage_report.csv" con righe del tipo: image_name,road_name,street_number,city,score
    generate_score()

    # Genera file "sorted_damage_report.csv" con righe analoghe al file precedente ma ordinate per score
    sort_score()

    # Prepara una lista di elementi contenenti il percorso dell'immagine e informazioni sull'indirizzo
    images_data = get_images_data()

    return total_damages, images_data


@inference_bp.route('/analizza_immagini', methods=['GET'])
def analizza():
    # Reperisco città di partenza e destinazione, servono solo ai fini di mostrare la barra "Percorso da ... a ..."
    partenza = session.get('partenza')
    destinazione = session.get('destinazione')

    # Fase d'inferenza del modello di ML sulle immagini salvate nella cartella images/input
    try:
        detect(source='images/input', weights='YOLOv7x_640.pt', imgsz=640, conf_thres=0.10, save_txt=True, view_img=False)
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    total_damages, images_data = genera_report()

    # Rendering del template con le immagini ordinate
    return render_template('inference.html', total_damages=total_damages, partenza=partenza, destinazione=destinazione, images_data=images_data, right_panel_flag=False)


@inference_bp.route('/output_images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(OUTPUT_IMAGE_FOLDER, filename)
