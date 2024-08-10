import streamlit as st
from PIL import Image
import pytesseract

# Set Tesseract OCR path (update this path according to your Tesseract installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image):
    try:
        # Perform OCR on the image
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return None

def save_to_text_file(text, filename="output.txt"):
    try:
        # Save the text to a text file
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        st.success(f"Text saved to {filename}")
    except Exception as e:
        st.error(f"Error saving text to file: {e}")

def main_page():
    st.title("Multi-Page Streamlit App")
    st.write("Select a page from the sidebar.")

def ocr_page():
    st.title("Optical Character Recognition")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform OCR on the image
        text_result = ocr_image(image)

        if text_result is not None:
            # Display the OCR result
            st.subheader("OCR Result:")
            st.text(text_result)

            # Add a button to save the text to a file
            if st.button("Download Text as File"):
                st.download_button(
                    label="Click to download",
                    data=text_result,
                    file_name="output.txt",
                    key="download_button",
                )

# Define the app pages
app_pages = {
    "Main Page": main_page,
    "OCR Page": ocr_page,
}

def main():
    # st.sidebar.title("Navigation")
    # selected_page = st.sidebar.radio("Go to", list(app_pages.keys()))

    # Execute the selected page function
    app_pages["OCR Page"]()

if __name__ == "__main__":
    main()
