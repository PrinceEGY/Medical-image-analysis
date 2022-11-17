import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


@st.experimental_singleton
def load_model(path):
    # Load large model
    model = tf.keras.models.load_model(path)
    return model


datasets = {
    'Retina diabetes': {'image': "images/retina.jpeg", 'model': 'Models h5/diabetic-eye.h5',
                        'size': (224, 224), 'classes': ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']},
    'Malaria Cells': {'image': "images/malaria.png", 'model': 'Models h5/malaria.h5',
                      'size': (120, 120), 'classes': ['Parasitized', 'Uninfected']},
}

st.set_page_config(
    page_title="Medical Images Analysis",
    page_icon="♥",
    layout='wide'
)


st.markdown("""
                <style>
                div {
                    font-size:1.02em !important;
                }
                label, input, textarea {
                    font-size:1.2em !important;
                }
                button{
                    font-size:1.8em !important;
                    font-weight: bold !important;
                }
                </style>
        """, unsafe_allow_html=True)

cols = st.columns([1, 3, 1, 4])
with cols[0]:
    st.image("images/doctor.png")

with cols[1]:
    st.image('images\heartbeat.png')
    with st.expander('Tool authors'):
        st.markdown("""
        **Authors: [Ahmed Mohsen](https://www.linkedin.com/in/AhmedMohsen-), [Yomna Ramdan]()**
        
        You can see the steps of preprocessing the data, building the models and evaluating it on GitHub repo [here](https://github.com/PrinceEGY/Medical-image-analysis).
        """)
    st.header('Medical Images Analysis')
    st.subheader('Are you wondering about your health condition?')
    st.markdown('''
    This application will help to analyze different medical imageries and predict whether there is a possibility of infection or not

    *Keep in mind that this results is not equivalent to a medical diagnosis!
    Doctors or patients CANNOT fully rely on it, but it can be used as an aid to onfirm the diagnosis, so if you have any problems, consult a human doctor.*

    To analyze your medical imagery, simply follow the steps bellow:
    1. Choose your type of image
    2. A sample of valid image of that type will show to you, your image should be of the same kind
    3. Upload your image
    4. Press the "Check" button and wait for results
    ''')

submit = False


def classifiy_img(image):
    image = Image.open(image)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # image sizing
    size = datasets[disease]['size']
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)/255.
    return image


with cols[3]:
    disease = st.selectbox("1. Choose your image", options=[
                           type for type in datasets])
    st.text('2.Sample of images')
    st.image(datasets[disease]['image'], width=300)
    file = st.file_uploader('3.Upload your image', type=["jpg", "png", "jpeg"])
    st.text('4.Check condition')
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        width:450px;
        height:75px;
        font-size:1.5em !important;
    }
        em{
        font-size:1.2em !important;
        font-style:normal;
        font-weight: bold;
        word-spacing: 2px;
    }
    </style>""", unsafe_allow_html=True)
    submit = st.button('Check imagery condition')
    st.text('5.Results')

    model = load_model(datasets[disease]['model'])

    if submit:
        preds = model.predict(np.expand_dims(classifiy_img(file), 0))

        st.markdown("""
        <style>
        strong {
            font-size:1.5em !important;
            color:#00FF00;
            text-indent: 50px;
        }
        </style>
        """, unsafe_allow_html=True)

        if len(datasets[disease]['classes']) == 2:
            pred = (1-preds[0][0])*100

            st.markdown(
                f"_Probability of {datasets[disease]['classes'][0]} is :_&nbsp;&nbsp; **{pred:.2f}%**")
            st.markdown(
                f"_Probability of {datasets[disease]['classes'][1]} is :_&nbsp;&nbsp; **{100-pred:.2f}%**")
        else:
            for idx, class_ in enumerate(datasets[disease]['classes']):
                pred = preds[0][idx]*100
                print(pred)
                st.markdown(
                    "_Probability of {0} is :_&nbsp;&nbsp; **{1:.2f}%**".format(class_, pred))
