import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import csv
import requests
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from model.utils import svuota_cartella

API_KEY = '' # Inserisci qui la tua API Key di Google Maps


# Funzione per eseguire la fase di inferenza sulle immagini scaricate
def detect(source='../images/input', weights='YOLOv7x_640.pt', imgsz=640, conf_thres=0.25, iou_thres=0.45,
           device='', view_img=True, save_txt=False, save_conf=False, nosave=False, classes=None,
           agnostic_nms=False, augment=False, project='images/output', filename='detected_damages'):
    save_img = not nosave and not source.endswith('.txt')  # Save images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(project)  # Use a single directory for saving
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Clear the output directory before adding new images
    svuota_cartella(save_dir)

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size

    if half:
        model.half()  # To FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Run once

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        sss = ""
        for i, det in enumerate(pred):  # Detections per image
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # To Path
            save_path = str(save_dir / p.name)  # Overwrite existing image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
            sss += p.stem + '.jpg' + ','

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    sss += str(int(cls) + 1) + ' '
                    for sz in xyxy:
                        sss += str(int(sz)) + ' '

                    # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)  # Overwrite the old image
                    print(f"The image with the result is saved in: {save_path}")

            with open(str(save_dir / filename), 'a') as fcsv:
                fcsv.write(sss + '\n')

            # Print time (inference + NMS)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


# Genera report in cui si contano quanti danni ci sono per ogni tipologia considerando tutte le immagini
def generate_report(file_path='images/output/detected_damages', output_image_path='static/damage_histogram.png'):
    # Dizionario con le chiavi predefinite 1, 2, 3, 4
    damage_count = { '1': 0, '2': 0, '3': 0, '4': 0 }

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("Il file non è stato trovato.")
        return damage_count
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        return damage_count

    for line in lines:
        line = line.strip()

        if line:
            parts = line.split(',')
            if len(parts) < 2:
                continue  # Ignora righe malformate

            # Il secondo elemento contiene le informazioni sui danni
            damage_class_info = parts[1].strip()

            # Split per ottenere le classi di danno
            damage_entries = damage_class_info.split()
            for i in range(0, len(damage_entries), 5):  # Saltiamo ogni 5 elementi (classe e coordinate)
                if i + 4 < len(damage_entries):
                    damage_class = damage_entries[i]
                    # Se la classe di danno è una delle chiavi predefinite, incrementiamo il suo contatore
                    if damage_class in damage_count:
                        damage_count[damage_class] += 1


    # Calcolare il numero totale di danni
    total_damages = sum(damage_count.values())

    # Creazione dell'istogramma
    if damage_count:
        # Ordinare le classi di danno (1, 2, 3, 4)
        damage_classes = list(damage_count.keys())
        counts = list(damage_count.values())

        # Colorazione personalizzata delle barre
        colors = ['#4285F4', '#34A853', '#EA4335', '#FBBC05']

        # Creazione dell'istogramma con colori personalizzati
        bars = plt.bar(damage_classes, counts, color=colors[:len(damage_classes)])

        # Aggiungere la legenda
        # Dizionario per mappare le classi ai nomi personalizzati
        label_mapping = {
            '1': 'D00 Longitudinal',
            '2': 'D10 Lateral',
            '3': 'D20 Alligator',
            '4': 'D40 Potholes and others'
        }

        # Aggiungere le etichette personalizzate alle barre
        for i, bar in enumerate(bars):
            damage_class = damage_classes[i]
            if damage_class in label_mapping:
                bar.set_label(label_mapping[damage_class])

        plt.xlabel('Tipologia di danno')
        plt.ylabel('Conteggio')
        plt.title('Distribuzione dei danni')
        plt.tight_layout()  # Ottimizza la disposizione per evitare sovrapposizioni

        # Aggiungere la legenda
        plt.legend()

        # Salva l'istogramma come immagine
        plt.savefig(output_image_path)
        print(f"Il grafico è stato salvato come immagine: {output_image_path}")

        # Chiude il grafico dopo il salvataggio
        plt.close()

    return total_damages


def get_address_details(latitude, longitude, api_key):
    # Geocode request to get address details
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={api_key}"
    geocode_response = requests.get(geocode_url)

    road_name, street_number, city = None, None, None
    if geocode_response.status_code == 200:
        geocode_results = geocode_response.json()
        if geocode_results['status'] == 'OK' and geocode_results['results']:
            # Extract address details
            for component in geocode_results['results'][0]['address_components']:
                if 'route' in component['types']:
                    road_name = component['long_name']
                if 'street_number' in component['types']:
                    street_number = component['long_name']
                if 'locality' in component['types']:
                    city = component['long_name']

    # Street View Metadata request to get image date
    street_view_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={latitude},{longitude}&key={api_key}"
    street_view_response = requests.get(street_view_url)

    photo_date = None
    if street_view_response.status_code == 200:
        street_view_results = street_view_response.json()
        if street_view_results['status'] == 'OK':
            # The date the photo was taken (if available)
            photo_date = street_view_results.get('date')
            inverted_date = "-".join(photo_date.split("-")[::-1])

    return road_name, street_number, city, inverted_date


# Genera file "damage_report.csv" con righe del tipo: image_name,road_name,street_number,city,score
def generate_score(file_path='images/output/detected_damages', output_csv='images/output/damage_report.csv', api_key=API_KEY):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("The file was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Prepare CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['image_name', 'road_name', 'street_number', 'city', 'photo_date', 'score'])

        # Process each line in the file
        for line in lines:
            line = line.strip()
            if line:  # Only process non-empty lines
                parts = line.split(',')

                if parts[1] != '':  # Check if the damage class info is present
                    image_name = parts[0]
                    if image_name.startswith("street_view_") and image_name.endswith(".jpg"):
                        # Try to parse the coordinates
                        try:
                            cleaned_name = image_name[len("street_view_"):-len(".jpg")]
                            lat_str, lon_str = cleaned_name.split('_')
                            latitude, longitude = float(lat_str), float(lon_str)
                            print(f"Processing image: {image_name} at coordinates: ({latitude}, {longitude})")

                            # Get address details using the Google Maps API
                            road_name, street_number, city, photo_date = get_address_details(latitude, longitude, api_key)

                            if road_name is not None:
                                # Extract damage data
                                damage_class_info = parts[1].strip()
                                damage_entries = damage_class_info.split()

                                # Calculate score
                                total_score = 0
                                valid_damage = True  # Flag to track valid damage entries

                                for i in range(0, len(damage_entries), 5):
                                    if i + 4 < len(damage_entries):
                                        # Parse damage data
                                        try:
                                            damage_type = int(damage_entries[i])  # Type of damage
                                            minx = int(damage_entries[i + 1])
                                            maxx = int(damage_entries[i + 2])
                                            miny = int(damage_entries[i + 3])
                                            maxy = int(damage_entries[i + 4])

                                            # Calculate surface area of the damage box
                                            surface_area = (maxx - minx) * (maxy - miny)
                                            score = surface_area
                                            total_score += score

                                            print(f"Damage type: {damage_type}, Bounding box: ({minx}, {maxx}, {miny}, {maxy}), Surface area: {surface_area}, Score: {score}")
                                        except ValueError:
                                            print(f"Invalid damage data format: {damage_entries[i:i+5]}")
                                            valid_damage = False  # Mark as invalid if any damage entry fails
                                            break  # Exit the damage parsing loop
                                    else:
                                        print(f"Incomplete damage entry: {damage_entries[i:]}")
                                        valid_damage = False
                                        break  # Exit if incomplete

                                # Write the result to the CSV file
                                if valid_damage and total_score > 0:
                                    csv_writer.writerow([image_name, road_name, street_number, city, photo_date, total_score])
                                    print(f"Written to CSV: ({image_name}, {road_name}, {street_number}, {city}, {photo_date}, {total_score})")
                                else:
                                    print(f"No valid damage entries found for image: {image_name}")

                        except ValueError:
                            print(f"Error parsing coordinates in image name: {image_name}")
                            continue  # Skip if parsing fails
                    else:
                        print(f"Invalid format for image name: {image_name}")
                else:
                    print(f"Skipping malformed line: {line}")

    print(f"Report generated: {output_csv}")


# Genera file "sorted_damage_report.csv" con righe analoghe al file precedente ma ordinate per score
def sort_score(input_csv='images/output/damage_report.csv', output_csv='images/output/sorted_damage_report.csv'):
    try:
        # Read the data from the input CSV file
        with open(input_csv, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Read the header
            data = list(csv_reader)  # Read the remaining rows

        # Convert score to integer for sorting
        for i in range(len(data)):
            data[i][5] = int(data[i][5])  # Assuming the score is in the sixth column

        # Create a list of tuples (original_index, row) for mapping
        indexed_data = list(enumerate(data))

        # Sort the indexed data by score (fourth column)
        sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1][5], reverse=True)  # Sort in descending order

        # Write the sorted data to the output CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)  # Write the header
            for _, row in sorted_indexed_data:
                csv_writer.writerow(row)  # Write the sorted rows

        # Create a mapping from old positions to new positions
        position_mapping = {old_index: new_index for new_index, (old_index, _) in enumerate(sorted_indexed_data)}

        print(f"Sorted report generated: {output_csv}")
        return position_mapping  # Return the mapping

    except FileNotFoundError:
        print("The input CSV file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Prepara una lista di elementi contenenti il percorso dell'immagine e informazioni sull'indirizzo
def get_images_data():
    images_data = []
    sorted_file_path = 'images/output/sorted_damage_report.csv'
    try:
        with open(sorted_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                # Assuming the row format is [image_name, road_name, street_number, city, photo_date, score]
                image_name, road_name, street_number, city, photo_date, score = row
                images_data.append({
                    'image_name': image_name,
                    'city': city,
                    'road_name': road_name,
                    'street_number': street_number,
                    'photo_date': photo_date,
                })
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file CSV: {e}")
        return 1
    return images_data
