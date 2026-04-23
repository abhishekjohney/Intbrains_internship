import os
import ollama
from pypdf import PdfReader
import docx

INPUT_FOLDER = "./documents"
OUTPUT_FOLDER = "./summaries"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)

def extract_text(file_path):
    extension = file_path.lower().split('.')[-1]
    
    try:
        if extension in ('txt', 'tex'):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as file:
                    return file.read()
                
        elif extension == 'pdf':
            text = ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
            
        elif extension == 'docx':
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        else:
            return None 
            
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return None

print(f"Scanning '{INPUT_FOLDER}' for documents...\n")

all_entries = os.listdir(INPUT_FOLDER)
files_in_folder = [
    name for name in all_entries
    if os.path.isfile(os.path.join(INPUT_FOLDER, name))
]

if not files_in_folder:
    print(f"No files found in '{INPUT_FOLDER}'.")
    print("Add .txt, .pdf, .docx, or .tex files and run again.")
    raise SystemExit(0)

print("Files available in documents folder:")
for index, filename in enumerate(files_in_folder, start=1):
    print(f"{index}. {filename}")

selected_filename = None
while selected_filename is None:
    user_input = input("\nEnter the file number to summarize: ").strip()

    if not user_input.isdigit():
        print("Please enter a valid number from the list.")
        continue

    choice = int(user_input)
    if choice < 1 or choice > len(files_in_folder):
        print("Choice out of range. Try again.")
        continue

    selected_filename = files_in_folder[choice - 1]

input_path = os.path.join(INPUT_FOLDER, selected_filename)
print(f"\n Extracting text from: {selected_filename}...")

document_text = extract_text(input_path)

if not document_text or not document_text.strip():
    print(f"⏭️ Cannot summarize {selected_filename} (Unsupported format or empty file)")
    raise SystemExit(0)

print(f" AI is summarizing: {selected_filename}...")

prompt = f"Please summarize the following document into a single shot paragraph:\n\n{document_text}"

try:
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])

    summary_text = response['message']['content']

    name_without_extension = selected_filename.rsplit('.', 1)[0]
    new_filename = f"{name_without_extension}_summary.txt"
    output_path = os.path.join(OUTPUT_FOLDER, new_filename)

    with open(output_path, "w", encoding="utf-8") as new_file:
        new_file.write(summary_text)

    print(f" Saved summary to: {output_path}\n")

except Exception as e:
    print(f" Error processing {selected_filename} with AI: {e}\n")

print("Selected document has been processed!")