import argparse
import re
import yaml
import time
import os
import json
import requests
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def split_html_by_sentence(html_str, max_chunk_size=2000):
    sentences = html_str.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += '. '
            current_chunk += sentence
    
    if current_chunk:
        chunks.append(current_chunk)

    # Remove dot from the beginning of first chunk
    if chunks and chunks[0].startswith('. '):
        chunks[0] = chunks[0][2:]

    # Add dot to the end of each chunk
    for i in range(len(chunks)):
        chunks[i] += '.'

    return chunks

def system_prompt(from_lang, to_lang):
    p  = "You are an %s-to-%s translator. " % (from_lang, to_lang)
    p += "Keep all special characters and HTML tags as in the source text. Return only %s translation." % to_lang
    return p

def create_batch_file(chunks, batch_file_path, from_lang, to_lang):
    with open(batch_file_path, 'w') as batch_file:
        for i, chunk in enumerate(chunks):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system_prompt(from_lang, to_lang)},
                        {"role": "user", "content": chunk}
                    ]
                }
            }
            batch_file.write(json.dumps(request) + '\n')

def upload_batch_file(client, batch_file_path):
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    return batch_input_file.id

def submit_batch(client, batch_input_file_id):
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "translation batch job"}
    )
    return response.id

def check_batch_status(client, batch_id):
    while True:
        response = client.batches.retrieve(batch_id)
        status = response.status
        if status == "completed":
            return response.output_file_id
        elif status == "failed":
            raise Exception(f"Batch {batch_id} failed: {response.error}")
        time.sleep(60)

def retrieve_batch_results(client, output_file_id):
    print(f"Downloading results from file {output_file_id}...")
    response = client.files.content(output_file_id)
    with open("response.jsonl", 'wb') as f:
        f.write(response.content)  # Write the binary content directly to a file

    results = []
    with open("response.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")
            content = data.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content", "")
            results.append((custom_id, content))
    
    return results    


def save_state(state, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        yaml.dump(state, f)

def load_state(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    return None

def translate(client, input_epub_path, output_epub_path, from_chapter=0, to_chapter=9999, from_lang='EN', to_lang='PL', checkpoint_file='translate_checkpoint.yml'):
    state = load_state(checkpoint_file)
    if not state:
        state = {
            'input_epub_path': input_epub_path,
            'output_epub_path': output_epub_path,
            'from_chapter': from_chapter,
            'to_chapter': to_chapter,
            'from_lang': from_lang,
            'to_lang': to_lang,
            'chapters': [],
            'initialized': False
        }

    book = epub.read_epub(input_epub_path)
    chapters = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    chapters_count = len(chapters)

    # Ensure the state has an entry for each chapter
    for i in range(chapters_count):
        if len(state['chapters']) <= i:
            state['chapters'].append({
                'translated_chunks': [],
                'remaining_chunks': [],
                'pending_batches': []
            })

    batch_ids = []
    for chapter_index, item in enumerate(chapters):
        if chapter_index < from_chapter or chapter_index > to_chapter:
            continue

        if state['chapters'][chapter_index]['pending_batches']:
            # If there are pending batches, add them to batch_ids for processing
            batch_ids.extend(state['chapters'][chapter_index]['pending_batches'])
        else:
            # If no pending batches, submit new batches
            print(f"Submitting batches for chapter {chapter_index + 1}/{chapters_count}...")
            soup = BeautifulSoup(item.content, 'html.parser')
            chunks = split_html_by_sentence(str(soup))
            batch_file_path = f"batch_chapter_{chapter_index}.jsonl"
            create_batch_file(chunks, batch_file_path, from_lang, to_lang)
            batch_input_file_id = upload_batch_file(client, batch_file_path)
            batch_id = submit_batch(client, batch_input_file_id)
            state['chapters'][chapter_index]['pending_batches'].append(batch_id)
            batch_ids.append(batch_id)

    state['initialized'] = True
    save_state(state, checkpoint_file)

       # Process all batches for each chapter separately
    for chapter_index, chapter_state in enumerate(state['chapters']):
        if chapter_index < from_chapter or chapter_index > to_chapter:
            continue

        chapter_results = []
        for batch_id in chapter_state['pending_batches']:
            output_file_id = check_batch_status(client, batch_id)
            chapter_results.extend(retrieve_batch_results(client, output_file_id))

        # Sort results by custom_id assuming the custom_id format is consistent within a chapter
        chapter_results.sort(key=lambda x: x[0])
        translated_text = ' '.join([result[1] for result in chapter_results])
        chapters[chapter_index].content = translated_text.encode('utf-8')

    epub.write_epub(output_epub_path, book, {})
    print("Translation completed and saved to:", output_epub_path)


def show_chapters(input_epub_path):
    book = epub.read_epub(input_epub_path)
    current_chapter = 1
    chapters_count = len([i for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT])

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            print(f"▶️  Chapter {current_chapter}/{chapters_count} ({len(item.content)} characters)")
            soup = BeautifulSoup(item.content, 'html.parser')
            chapter_beginning = soup.text[0:250]
            chapter_beginning = re.sub(r'\n{2,}', '\n', chapter_beginning)
            print(chapter_beginning + "\n\n")
            current_chapter += 1

def resume_translation(client, checkpoint_file='translate_checkpoint.yml'):
    state = load_state(checkpoint_file)
    if not state:
        print("No checkpoint file found. Please start a new translation.")
        return

    input_epub_path = state['input_epub_path']
    output_epub_path = state['output_epub_path']
    from_chapter = state['from_chapter']
    to_chapter = state['to_chapter']
    from_lang = state['from_lang']
    to_lang = state['to_lang']

    translate(client, input_epub_path, output_epub_path, from_chapter, to_chapter, from_lang, to_lang, checkpoint_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='App to translate or show chapters of a book.')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation.')

    parser_translate = subparsers.add_parser('translate', help='Translate a book.')
    parser_translate.add_argument('--input', required=True, help='Input file path.')
    parser_translate.add_argument('--output', required=True, help='Output file path.')
    parser_translate.add_argument('--config', required=True, help='Configuration file path.')
    parser_translate.add_argument('--from-chapter', type=int, help='Starting chapter for translation.', default=0)
    parser_translate.add_argument('--to-chapter', type=int, help='Ending chapter for translation.', default=9999)
    parser_translate.add_argument('--from-lang', help='Source language.', default='EN')
    parser_translate.add_argument('--to-lang', help='Target language.', default='PL')

    parser_show = subparsers.add_parser('show-chapters', help='Show the list of chapters.')
    parser_show.add_argument('--input', required=True, help='Input file path.')

    parser_resume = subparsers.add_parser('resume', help='Resume translation from a checkpoint.')

    args = parser.parse_args()

    if args.mode == 'translate':
        config = read_config(args.config)
        from_lang = args.from_lang
        to_lang = args.to_lang
        openai_client = OpenAI(api_key=config['openai']['api_key'])

        translate(openai_client, args.input, args.output, args.from_chapter, args.to_chapter, from_lang, to_lang)

    elif args.mode == 'show-chapters':
        show_chapters(args.input)

    elif args.mode == 'resume':
        config = read_config('config.yaml')  # Assuming the config file path is fixed for resuming
        openai_client = OpenAI(api_key=config['openai']['api_key'])
        resume_translation(openai_client)

    else:
        parser.print_help()
