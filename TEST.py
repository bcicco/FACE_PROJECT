import json
import csv

def process_json_to_csv(json_file, label_value):
    # Load JSON data from file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Write data to CSV file
    with open('facedata.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for entry in data:
            if entry:  # Check if entry is not None
                # Append an empty column for the label
                entry.append(label_value)
                writer.writerow(entry)

# Clear the existing content of the output file
with open('facedata.csv', 'w', newline=''):
    pass

# Process the first JSON file with label 0
process_json_to_csv('nondrowsy.json', 0)

# Process the second JSON file with label 1
process_json_to_csv('output.json', 1)
