import pathlib
import io
import PyPDF2
import docx
import docx2txt
import nltk
import re
from transformers import (
    pipeline, 
    BartTokenizer, 
    BartForConditionalGeneration, 
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer
    )
from heapq import nlargest
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

models_directory = "D:/AIModels/"

# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def extract_text_from_doc(file_path):
    try:
        if file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.doc'):
            text = docx2txt.process(file_path)
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def generate_text_summary(text, chunk_size: int = 1000, sentence_cut_percentage: float = 25):
    percentage_kept = (100 - sentence_cut_percentage) / 100
    sentences = sent_tokenize(text)
    num_sentences = int(len(sentences) * percentage_kept)

    #load the AI model
    tokenizer = BartTokenizer.from_pretrained(models_directory + "bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained(models_directory + "bart-large-cnn")

    # Preprocess the text
    stopwords_list = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords_list]

    # Calculate word frequency
    frequency = FreqDist(words)

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in frequency:
                if len(sentence.split(' ')) < 30:  # Ignore very long sentences
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = frequency[word]
                    else:
                        sentence_scores[sentence] += frequency[word]

    # Get the top 'num_sentences' sentences with highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    abstractedText = ' '.join(summary_sentences)

    # Split the text into chunks (customize chunk size based on your requirements)
    chunks = [abstractedText[i:i+chunk_size] for i in range(0, len(abstractedText), chunk_size)]

    # Summarize each chunk and concatenate the summaries
    summaries = []
    for chunk in chunks:
        inputs = tokenizer([chunk], return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, num_beams=2, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        #summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    final_summary = " ".join(summaries)

    return final_summary

def extract_names(text):
    model = AutoModelForTokenClassification.from_pretrained(models_directory + "bert-large-ner")
    tokenizer = AutoTokenizer.from_pretrained(models_directory + "bert-large-ner")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    sentences = nltk.sent_tokenize(text)
    named_entities = ner_pipeline(sentences)
    full_names = []

    for sentence_entities in named_entities:
        current_name = ''
        for entity in sentence_entities:
            if entity['entity'] == 'B-PER':
                if current_name:
                    full_names.append(current_name.strip())
                    current_name = ''
                current_name = entity['word']
            elif entity['entity'] == 'I-PER':
                current_name += ' ' + entity['word']

        if current_name:
            full_names.append(current_name.strip())

    names_string = ', '.join(full_names)
    return names_string

def ask_question(question, context, modelChoice):
    QA_input = {
        'question': question,
        'context': context
    }

    if modelChoice == "roberta-base-squad2":
        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "roberta-base-squad2")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    elif modelChoice == "bert-base-cased-squad2":
        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "bert-base-cased-squad2")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "bert-base-cased-squad2")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    elif modelChoice == "bert-large-uncased-whole-word-masking-finetuned-squad":
        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "bert-large-uncased-whole-word-masking-finetuned-squad")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    elif modelChoice == "ensemble":
        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "roberta-base-squad2")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
        
        resultA = qa_pipeline(QA_input)

        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "bert-base-cased-squad2")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "bert-base-cased-squad2")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
        
        resultB = qa_pipeline(QA_input)

        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "bert-large-uncased-whole-word-masking-finetuned-squad")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

        resultC = qa_pipeline(QA_input)
    
        results = [resultA, resultB, resultC]
    
        best_result = max(results, key=lambda result: result['score'])

        answerContext = extract_answer_context(context, best_result['start'], best_result['end'])

        best_result['answerContext'] = answerContext

        return best_result
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(models_directory + "roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained(models_directory + "roberta-base-squad2")
        qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

    result = qa_pipeline(QA_input)

    answerContext = extract_answer_context(context, result['start'], result['end'])

    result['answerContext'] = answerContext

    return result

def extract_answer_context(full_text, start_idx, end_idx):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(full_text)

    # Split each sentence based on newlines
    sentences_with_newlines = [re.split(r'\n', sentence) for sentence in sentences]

    # Flatten the list to get sentences with newlines as separate sentences
    sentences = [sentence for sublist in sentences_with_newlines for sentence in sublist]

    # Find the sentence containing the answer
    answer_sentence = None
    for sentence in sentences:
        if start_idx >= full_text.find(sentence) and end_idx <= full_text.find(sentence) + len(sentence):
            answer_sentence = sentence
            break

    return answer_sentence