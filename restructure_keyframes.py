import os

# Specify the range of directories to search
start_folder = 48
end_folder = 59

# Open the final file for writing
path_keyframe = "/home/jonathan/Downloads/YCB-Video-Base/keyframes/"
final_file_path = "/home/jonathan/Downloads/"

# Check if the keyframe.txt file already exists

with open(final_file_path +" keyframe.txt", "w") as final_file:
    # Iterate over the directories
    for folder_number in range(start_folder, end_folder + 1):
        # Create the directory path
        directory = f"{str(folder_number).zfill(4)}"

        # Create the file path
        file_path = os.path.join(path_keyframe + directory, "keyframes.txt")

        # Check if the file exists
        if os.path.exists(file_path):
            # Open the file for reading
            with open(file_path, "r") as keyframes_file:
                # Read the numbers from each line
                for line in keyframes_file:
                    # Remove leading/trailing whitespace and split the line
                    number = line.strip()

                    # Write the formatted number to the final file
                    final_file.write(f"{str(folder_number).zfill(4)}/{number.zfill(6)}\n")
        else:
            print(f"File not found: {file_path}")