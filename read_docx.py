import sys
import zipfile
import xml.etree.ElementTree as ET
import os

PATH = r"c:\Projects\bot_platform\docs\theodor_ai\Прмпт Федор АИ.docx"
OUTPUT = r"c:\Projects\bot_platform\extracted_text.txt"

def read_docx_xml(path):
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            text_parts = []
            # Iterate over all elements to capture everything in order
            for elem in tree.iter():
                # If it's a paragraph start, maybe add newline? 
                # Actually, usually we want newline at the end of p.
                # But iter() is depth-first.
                # Let's just check for 'p' and 't'.
                pass
            
            # Better approach: iterate paragraphs, then content
            # But to be safe against nesting, let's just do a flat iteration and insert newlines for 'p'
            
            # Re-parsing to be sure
            text_parts = []
            for elem in tree.iter():
                if elem.tag.endswith('}p'):
                    text_parts.append('\n')
                elif elem.tag.endswith('}t'):
                    if elem.text:
                        text_parts.append(elem.text)
                elif elem.tag.endswith('}tab'):
                    text_parts.append('\t')
                elif elem.tag.endswith('}br'):
                    text_parts.append('\n')
            
            return ''.join(text_parts)
    except Exception as e:
        return f"Error reading docx via XML: {e}"

if __name__ == "__main__":
    content = read_docx_xml(PATH)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done")
