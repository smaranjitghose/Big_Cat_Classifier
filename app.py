import time

import streamlit as st
from PIL import Image, ImageOps

from src import big_cat_classifier


def main():
    st.set_page_config(
        page_title="Big Cat Classifier",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    banner_img = Image.open("./assets/banner_img.png")
    st.image(banner_img)
    st.title("Coded with ❤️ by Smaranjit Ghose")
    st.text("")
    st.text("")
    st.text("")

    uploaded_file = st.file_uploader("Choose an image..", type=["jpg", "png", "jpeg"])

    if st.button("Predict"):
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.subheader("Your Image:")
                st.image(img)
                st.write("")
                st.write("")

                with st.spinner("Our AI forest officer has started analyzing...."):
                    label = big_cat_classifier.classifier(uploaded_file)
                    time.sleep(5)
                    st.success(f"We think this is an image of a {label}")

            except:
                st.error("We apologize something went wrong 🙇🏽‍♂️")
        else:
            st.error("Can you please upload an image 🙇🏽‍♂️")


if __name__ == "__main__":
    main()
