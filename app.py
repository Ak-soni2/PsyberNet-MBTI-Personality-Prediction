import streamlit as st
import nbformat

# Function to read the Jupyter Notebook
def load_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    return notebook

# Display the contents of the notebook
def display_notebook(notebook):
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            st.markdown(cell.source)
        elif cell.cell_type == 'code':
            st.code(cell.source)
        elif cell.cell_type == 'raw':
            st.write(cell.source)

# Load the Jupyter Notebook
notebook_path = 'IML Project.ipynb'  # Adjust the path if necessary
notebook = load_notebook(notebook_path)

# Display the notebook content in the Streamlit app
st.title("Contents of IML Project Notebook")
display_notebook(notebook)
