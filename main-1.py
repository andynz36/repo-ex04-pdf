# pip install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

file_path = "운수.pdf"
loader = PyPDFLoader(file_path)

pages = loader.load_and_split()

# print(pages)
# print(pages[0])
# print(pages[3])