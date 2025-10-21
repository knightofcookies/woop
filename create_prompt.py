import os
import argparse
import pyperclip

def build_context_from_directory(directory_path, excluded_dirs=None):
    if excluded_dirs is None:
        excluded_dirs = ['venv', 'env', '.env', '.git', 'models']

    full_context = ""
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return full_context

    for root, dirs, files in os.walk(directory_path):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if file.endswith(".py") or file.endswith(".yaml") :
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                
                header = f"# --- File: {relative_path} ---\n"
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    full_context += header + content + "\n\n"
                except Exception as e:
                    full_context += f"# --- Could not read file: {relative_path} due to {e} ---\n\n"
                    
    return full_context

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="A script to recursively gather Python code from a directory to create a context for an LLM prompt.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("directory", help="The path to the directory you want to scan.")
    
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-f", "--file", dest="filename",
                              help="Save the output to a specified file.\nExample: -f output.txt")
    output_group.add_argument("-c", "--copy", action="store_true",
                              help="Copy the output directly to the clipboard.\nRequires the 'pyperclip' library (pip install pyperclip).")

    args = parser.parse_args()
    
    code_context = build_context_from_directory(args.directory)
    
    if not code_context:
        print("No Python files were found or the directory does not exist.")
        return

    if args.filename:
        try:
            with open(args.filename, 'w', encoding='utf-8') as f:
                f.write(code_context)
            print(f"Successfully saved the code context to '{args.filename}'")
        except IOError as e:
            print(f"Error writing to file '{args.filename}': {e}")
            
    elif args.copy:
        try:
            pyperclip.copy(code_context)
            print("Code context has been copied to your clipboard successfully!")
        except pyperclip.PyperclipException:
            print("\n---\nError: Could not copy to clipboard.")
            print("Please make sure you have a clipboard utility installed, or install pyperclip with 'pip install pyperclip'.")
            print("Displaying context in the console instead:\n---\n")
            print(code_context)
            
    else:
        print("\n--- Generated Code Context ---\n")
        print(code_context)

if __name__ == "__main__":
    main()
