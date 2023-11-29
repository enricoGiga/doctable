import os
import tempfile

import pandas as pd
import streamlit as st

from src.doctable import Doctable

doctable = Doctable()
# Create a sidebar for file upload
st.sidebar.title("Upload PDF or Image")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or image file",
                                         type=["pdf", "png", "jpg", "jpeg"])

# Add a button for processing the file
if st.button('Process File'):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix="." + uploaded_file.type.split('/')[
                                             1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        # Process the uploaded file with Doctable

        with st.spinner('Processing...'):
            pages = doctable.table_extraction(tmp_path)

        # Display the results
        for i, page in enumerate(pages):
            for j, table in enumerate(page.tables):
                # Convert the HTML to a DataFrame
                df = pd.read_html(table.recognition_results["html"])[0]

                # Display the DataFrame in the Streamlit application
                st.write(f'Page {i + 1} Table {j + 1}:')
                st.dataframe(df)

        # Delete the temporary file
        os.remove(tmp_path)
    else:
        st.write("No file uploaded. Please upload a file.")
