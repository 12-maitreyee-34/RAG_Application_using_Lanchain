import os
from RAG.ingestion import pdf_to_document, save_document, save_session

# 1. Database is managed by Alembic now
print("Make sure you have run Alembic migrations!")

# 2. Use the Data folder as you requested
data_dir = "Data"

if not os.path.exists(data_dir):
    print(f"Error: Folder '{data_dir}' not found!")
else:
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDFs found in '{data_dir}' folder!")
    else:
        doc_ids = []
        for filename in pdf_files:
            path = os.path.join(data_dir, filename)
            print(f"\nProcessing: {filename}...")
            
            try:
                # Convert PDF to our Document model
                doc = pdf_to_document(path)
                
                # Save to database
                save_document(doc)
                doc_ids.append(doc.doc_id)
                
                print(f"Saved: {doc.title}")
                print(f"Sections found: {len(doc.sections)}")
                for s in doc.sections:
                    print(f"    - {s['heading']:<25} ({len(s['content'].split())} words)")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # 3. Create a session for the uploaded papers
        if doc_ids:
            try:
                session_id = save_session(doc_ids)
                print(f"\n SUCCESS! All papers processed.")
                print(f"Session ID: {session_id}")
            except Exception as e:
                print(f"Error creating session: {e}")
