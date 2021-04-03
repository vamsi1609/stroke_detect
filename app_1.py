import streamlit as st
from apps import home ,result
from multiapp import MultiApp

app = MultiApp()
app.add_app("Home", home.app)
app.add_app("Result",result.app)
app.run()

