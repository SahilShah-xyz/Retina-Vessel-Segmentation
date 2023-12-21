import re
import csv


def extract_data(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if "Epoch:" in lines[i]:
                epoch_line = lines[i]
                train_loss_line = lines[i + 1]
                val_loss_line = lines[i + 2]
                i += 4

                epoch = int(epoch_line.split(":")[1].strip().split()[0])
                train_loss = float(train_loss_line.split(":")[1].strip())
                val_loss = float(val_loss_line.split(":")[1].strip())
                data.append((epoch, train_loss, val_loss))
            i += 1
    return data


def write_to_csv(output_file, data):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epoch", "Train Loss", "Val Loss"])
        csvwriter.writerows(data)


def main():
    input_file = "./att_unet_drive_200.txt"
    output_file = "att_unet_drive_200.csv"

    extracted_data = extract_data(input_file)
    # The code `print(extracted_data)` is printing the extracted data from the input file. It is used to
    # verify that the data extraction is working correctly.
    write_to_csv(output_file, extracted_data)


if __name__ == "__main__":
    main()
