import re

def add_waw_to_arabic_words(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    arabic_pattern = re.compile(r'^[\u0600-\u06FF]+')  # Matches lines starting with Arabic letters

    for line in lines:
        stripped_line = line.strip()
        if arabic_pattern.match(stripped_line):  
            updated_lines.append(stripped_line)         # Original name
            updated_lines.append("و" + stripped_line)  # Name with "و"
        else:
            updated_lines.append(stripped_line)  # Keep headers and other text as is

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")  # Preserve formatting

    print(f"Processed text saved to {output_file}")

# Example usage
add_waw_to_arabic_words("isnad_words_no_waw.txt", "isnad_words.txt")
